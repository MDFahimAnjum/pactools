import numpy as np
from scipy.signal import hilbert

from .utils.maths import compute_n_fft
from .utils.carrier import Carrier
from .utils.fir import BandPassFilter
import scipy.signal as signal

def multiple_band_pass(sigs, fs, frequency_range, bandwidth, n_cycles=None,
                       filter_method='eegfilt'):
    """
    Band-pass filter the signal at multiple frequencies

    Parameters
    ----------
    sigs : array, shape (n_epochs, n_points)
        Input array to filter

    fs : float
        Sampling frequency

    frequency_range : float, list, or array, shape (n_frequencies, )
        List of center frequency of bandpass filters.

    bandwidth : float
        Bandwidth of the bandpass filters.
        Use it to have a constant bandwidth for all filters.
        Should be None if n_cycles is not None.

    n_cycles : float
        Number of cycles of the bandpass filters.
        Use it to have a bandwidth proportional to the center frequency.
        Should be None if bandwidth is not None.

    filter_method : string, in {'mne', 'pactools'}
        Method to bandpass filter.
        - 'pactools' uses internal wavelet-based bandpass filter.
        - 'mne' uses mne.filter.band_pass_filter in MNE-python package
        - 'eegfilt' uses implimentation of FIR filter of eegfilt.m script from EEGLab. (default)

    Returns
    -------
    filtered : array, shape (n_frequencies, n_epochs, n_points)
        Bandpass filtered version of the input signals
    """
    fixed_n_cycles = n_cycles

    sigs = np.atleast_2d(sigs)
    n_fft = compute_n_fft(sigs)
    n_epochs, n_points = sigs.shape

    frequency_range = np.atleast_1d(frequency_range)
    n_frequencies = frequency_range.shape[0]

    if filter_method == 'carrier':
        fir = Carrier()

    filtered = np.zeros((n_frequencies, n_epochs, n_points),
                        dtype=np.complex128)

    # eegfilt parameters
    minfac = 3  # example multiplier, same as in MATLAB
    min_filtorder = 15  # minimum filter length

    for jj, frequency in enumerate(frequency_range):

        if frequency <= 0:
            raise ValueError("Center frequency for bandpass filter should"
                             "be non-negative. Got %s." % (frequency, ))
        # evaluate the number of cycle for this bandwidth and frequency
        if fixed_n_cycles is None:
            n_cycles = 1.65 * frequency / bandwidth

        # --------- with mne.filter.band_pass_filter
        if filter_method == 'mne':
            from mne.filter import band_pass_filter
            for ii in range(n_epochs):
                low_sig = band_pass_filter(
                    sigs[ii, :], Fs=fs, Fp1=frequency - bandwidth / 2.0,
                    Fp2=frequency + bandwidth / 2.0,
                    l_trans_bandwidth=bandwidth / 4.0,
                    h_trans_bandwidth=bandwidth / 4.0, n_jobs=1, method='iir')

                filtered[jj, ii, :] = hilbert(low_sig, n_fft)[:n_points]

        # --------- with pactools.utils.Carrier (deprecated)
        elif filter_method == 'carrier':
            for ii in range(n_epochs):
                fir.design(fs, frequency, n_cycles, None, zero_mean=True)
                low_sig = fir.direct(sigs[ii, :])
                filtered[jj, ii, :] = hilbert(low_sig, n_fft)[:n_points]

        # --------- with pactools.utils.BandPassFilter
        elif filter_method == 'pactools':
            fir = BandPassFilter(fs, fc=frequency, n_cycles=n_cycles,
                                 bandwidth=None, zero_mean=True,
                                 extract_complex=True)
            low_sig, low_sig_imag = fir.transform(sigs)
            filtered[jj, :, :] = low_sig + 1j * low_sig_imag
        
        # --------- with eegfilt
        elif filter_method == 'eegfilt':
            locutoff = frequency - bandwidth / 2.0  # low cutoff frequency (replace with actual value)
            hicutoff = frequency + bandwidth / 2.0  # high cutoff frequency (replace with actual value)

            # Calculate the filter order
            filtorder = int(minfac * (fs // locutoff))  # Use integer division
            if filtorder < min_filtorder:
                filtorder = min_filtorder
            
            # Design the FIR filter (using the window method, similar to fir1 in MATLAB)
            filtwts = signal.firwin(filtorder + 1, [locutoff, hicutoff], pass_zero=False, fs=fs)
            # Apply zero-phase filtering
            low_sig = signal.filtfilt(filtwts, 1, sigs, axis=1)
            # Apply Hilbert transform along each row
            analytic_sigs = hilbert(low_sig, axis=1)  # Apply Hilbert transform along rows
            analytic_sigs =analytic_sigs[:,:n_points]
            filtered[jj, :, :]=analytic_sigs            

    return filtered
