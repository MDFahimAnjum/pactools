import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from scipy.signal import hilbert
from mne.filter import band_pass_filter

from .dar_model.dar import DAR
from .utils.progress_bar import ProgressBar
from .utils.spectrum import crop_for_fast_hilbert
from .utils.carrier import Carrier
from .utils.maths import norm, argmax_2d
from .plot_comodulogram import plot_comodulograms
from .preprocess import extract


def multiple_band_pass(sigs, fs, frequency_range, bandwidth,
                       n_cycles=None, method=1):
    """
    Band-pass filter the signal at multiple frequencies
    """
    fixed_n_cycles = n_cycles

    sigs = np.atleast_2d(sigs)
    sigs = crop_for_fast_hilbert(sigs)
    n_epochs, n_points = sigs.shape

    frequency_range = np.atleast_1d(frequency_range)
    n_frequencies = frequency_range.shape[0]

    if method == 1:
        fir = Carrier()

    filtered = np.zeros((n_frequencies, n_epochs, n_points),
                        dtype=np.complex128)
    for ii in range(n_epochs):
        for jj, frequency in enumerate(frequency_range):
            # evaluate the number of cycle for this bandwidth and frequency
            if fixed_n_cycles is None:
                n_cycles = 1.65 * frequency / bandwidth

            # 0--------- with mne.filter.band_pass_filter
            if method == 0:
                low_sig = band_pass_filter(
                    sigs[ii, :], Fs=fs,
                    Fp1=frequency - bandwidth / 2.0,
                    Fp2=frequency + bandwidth / 2.0,
                    l_trans_bandwidth=bandwidth / 4.0,
                    h_trans_bandwidth=bandwidth / 4.0,
                    n_jobs=1, method='iir')

            # 1--------- with pactools.Carrier
            if method == 1:
                fir.design(fs, frequency, n_cycles, None, zero_mean=True)
                low_sig = fir.direct(sigs[ii, :])

            # common to the two methods
            filtered[jj, ii, :] = hilbert(low_sig)

    return filtered


def _comodulogram(filtered_low, filtered_high, mask, method, fs, n_surrogates,
                  progress_bar, draw_phase):
    """
    Compute the comodulogram for empirical metrics.
    """
    rng = RandomState(42)

    # The modulation index is only computed where mask is True
    if mask is not None:
        mask = crop_for_fast_hilbert(mask)
        filtered_low = filtered_low[:, mask == 1]
        filtered_high = filtered_high[:, mask == 1]
    else:
        filtered_low = filtered_low.reshape(filtered_low.shape[0], -1)
        filtered_high = filtered_high.reshape(filtered_high.shape[0], -1)

    n_low, n_points = filtered_low.shape
    n_high, _ = filtered_high.shape

    # phase of the low frequency signals
    for i in range(n_low):
        filtered_low[i] = np.angle(filtered_low[i])
    filtered_low = np.real(filtered_low)

    # amplitude of the high frequency signals
    norm_a = np.zeros(n_high)
    for j in range(n_high):
        filtered_high[j] = np.abs(filtered_high[j])
        if method == 'ozkurt':
            norm_a[j] = norm(filtered_high[j])
    filtered_high = np.real(filtered_high)

    # Calculate the modulation index for each couple
    MI = np.zeros((n_low, n_high))
    exp_phase = None
    for i in range(n_low):
        if method != 'tort':
            exp_phase = np.exp(1j * filtered_low[i])

        for j in range(n_high):
            MI[i, j] = _one_modulation_index(
                amplitude=filtered_high[j], phase=filtered_low[i],
                exp_phase=exp_phase, norm_a=norm_a[j], method=method,
                n_points=n_points, fs=fs, n_surrogates=n_surrogates,
                random_state=rng, draw_phase=draw_phase)

        if progress_bar:
            progress_bar.update_with_increment_value(1)

    return MI


def _one_modulation_index(amplitude, phase, exp_phase, norm_a, method,
                          n_points, fs, n_surrogates, random_state,
                          draw_phase):
    if method == 'ozkurt':
        # Modulation index as in Ozkurt 2011
        MI = np.abs(np.mean(amplitude * exp_phase))

        MI /= norm_a
        MI *= np.sqrt(n_points)

    elif method == 'canolty':
        # Modulation index as in Canolty 2006
        MI = np.abs(np.mean(amplitude * exp_phase))

        if n_surrogates > 0:
            # compute surrogates MIs
            MI_surr = np.empty(n_surrogates)
            for s in range(n_surrogates):
                shift = random_state.randint(fs, n_points - fs)
                exp_phase_s = np.roll(exp_phase, shift)
                exp_phase_s *= amplitude
                MI_surr[s] = np.abs(np.mean(exp_phase_s))

            MI -= np.mean(MI_surr)
            MI /= np.std(MI_surr)

    elif method == 'tort':
        # Modulation index as in Tort 2010

        # mean amplitude distribution along phase bins
        n_bins = 18 + 2
        while n_bins > 0:
            n_bins -= 2
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            bin_indices = np.digitize(phase, phase_bins) - 1
            amplitude_dist = np.zeros(n_bins)
            for b in range(n_bins):
                selection = amplitude[bin_indices == b]
                if selection.size == 0:  # no sample in that bin
                    continue
                amplitude_dist[b] = np.mean(selection)
            if np.any(amplitude_dist == 0):
                continue

            # Kullback-Leibler divergence of the distribution vs uniform
            amplitude_dist /= np.sum(amplitude_dist)
            divergence_kl = np.sum(amplitude_dist *
                                   np.log(amplitude_dist * n_bins))

            MI = divergence_kl / np.log(n_bins)
            break

        if draw_phase:
            phase_bins = 0.5 * (phase_bins[:-1] + phase_bins[1:]) / np.pi * 180
            plt.plot(phase_bins, amplitude_dist, '.-')
            plt.plot(phase_bins, np.ones(n_bins) / n_bins, '--')
            plt.ylim((0, 2. / n_bins))
            plt.xlim((-180, 180))
            plt.ylabel('Normalized mean amplitude')
            plt.xlabel('Phase (in degree)')
            plt.title('Tort index: %.3f' % MI)

    else:
        raise(ValueError,
              "Unknown method for modulation index: Got '%s' instead "
              "of one in ('canolty', 'ozkurt', 'tort')" % method)

    return MI


def comodulogram(fs, low_sig, high_sig=None, mask=None,
                 low_fq_range=np.linspace(1.0, 10.0, 50),
                 high_fq_range=np.linspace(5.0, 150.0, 60),
                 low_fq_width=0.5,
                 high_fq_width=10.0,
                 method='tort',
                 n_surrogates=100,
                 draw=False, save_name=None,
                 vmin=None, vmax=None,
                 progress_bar=True,
                 draw_phase=False):
    """
    Compute the comodulogram for Phase Amplitude Coupling (PAC).

    Parameters
    ----------
    fs : float,
        Sampling frequency

    low_sig : array, shape (n_epochs, n_points)
        Input data for the phase signal

    high_sig : array or None, shape (n_epochs, n_points)
        Input data for the amplitude signal.
        If None, we use low_sig for both signals

    mask : array or list of array or None, shape (n_epochs, n_points)
        The PAC is only evaluated with the unmasked element of low_sig and
        high_sig. Masking is done after filtering and Hilbert transform.
        If a list is given, the filtering is done only once and the
        comodulogram is computed on each mask.

    low_fq_range : array or list
        List of filtering frequencies (phase signal)

    high_fq_range : array or list
        List of filtering frequencies (amplitude signal)

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    high_fq_width : float
        Bandwidth of the band-pass filter (amplitude signal)

    method : string in ('ozkurt', 'canolty', 'tort')
        Modulation index method,

    n_surrogates : int
        Number of surrogates computed in 'canolty's method

    draw : boolean
        If True, plot the comodulogram

    vmin, vmax : float or None
        If not None, it define the min/max value of the plot

    progress_bar : boolean
        If True, a progress bar is shown in stdout

    draw_phase : boolean
        If True, plot the phase distribution in 'tort' index

    Return
    ------
    comod : array, shape (len(low_fq_range), len(high_fq_range))
        Comodulogram for each couple of frequencies.
        If a list of mask is given, it returns a list of comodulograms.
    """
    # convert to numpy array
    low_fq_range = np.asarray(low_fq_range)
    high_fq_range = np.asarray(high_fq_range)

    mask_is_list = isinstance(mask, list)
    if not mask_is_list:
        mask = [mask]

    if method in ('ozkurt', 'canolty', 'tort'):
        if high_sig is None:
            high_sig = low_sig

        # compute a number of band-pass filtered and Hilbert filtered signals
        filtered_high = multiple_band_pass(high_sig, fs,
                                           high_fq_range, high_fq_width)
        filtered_low = multiple_band_pass(low_sig, fs,
                                          low_fq_range, low_fq_width)

        if progress_bar:
            progress_bar = ProgressBar('comodulogram: %s' % method,
                                       max_value=low_fq_range.size * len(mask))
        comod_list = []
        for this_mask in mask:
            comod = _comodulogram(filtered_low, filtered_high, this_mask,
                                  method, fs, n_surrogates, progress_bar,
                                  draw_phase)
            comod_list.append(comod)

    elif isinstance(method, DAR):
        comod_list = driven_comodulogram(fs=fs, low_sig=low_sig,
                                         high_sig=high_sig,
                                         mask=mask, model=method,
                                         low_fq_range=low_fq_range,
                                         low_fq_width=low_fq_width,
                                         high_fq_range=high_fq_range,
                                         progress_bar=progress_bar)
    else:
        raise(ValueError, 'unknown method: %s' % method)

    if draw:
        plot_comodulograms(comod_list, fs, low_fq_range)

    if not mask_is_list:
        return comod_list[0]
    else:
        return comod_list


def driven_comodulogram(fs, low_sig, high_sig, mask, model, low_fq_range,
                        high_fq_range, low_fq_width, method='minmax',
                        fill=4, ordar=12, enf=50., random_noise=None,
                        normalize=True, whitening=None,
                        progress_bar=True):
    """
    Compute the comodulogram with a DAR model.

    Parameters
    ----------
    fs : float,
        Sampling frequency

    low_sig : array, shape (n_epochs, n_points)
        Input data for the phase signal

    high_sig : array or None, shape (n_epochs, n_points)
        Input data for the amplitude signal.
        If None, we use low_sig for both signals

    mask : array or list of array or None, shape (n_epochs, n_points)
        The PAC is only evaluated with the unmasked element of low_sig and
        high_sig. Masking is done after filtering and Hilbert transform.
        If a list is given, the filtering is done only once and the
        comodulogram is computed on each mask.

    model : DAR instance
        DAR model to be used for the comodulogram

    low_fq_range : array or list
        List of filtering frequencies (phase signal)

    high_fq_range : array or list
        List of filtering frequencies (amplitude signal). This is not used for
        filtering since DAR models do not need filtering of high frequencies.
        This is only used to interpolate the spectrum, in order to
        match the results of ``comodulogram``.

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    method : string in ('firstlast', 'minmax')
        Modulation index method,

    fill : int in (0, 1, 2, 3, 4)
        Method to fill the spectral gap when removing the low frequencies

    ordar : int
        Order of the AR model used for whitening

    enf : float
        Electric network frequency, that will be removed

    random_noise : array or None, shape (n_points)
        Noise to be used to in the filling strategy

    normalize : boolean
        If True, the filtered signal is normalized

    whitening : boolean
        If True, the filtered signal is whitened

    progress_bar : boolean
        If True, a progress bar is shown in stdout

    Return
    ------
    comod : array, shape (len(low_fq_range), len(high_fq_range))
        Comodulogram for each couple of frequencies
    """
    if high_sig is None:
        sigs = low_sig
    else:
        sigs = np.r_[low_sig, high_sig]
        n_epochs = low_sig.shape[0]

    sigs = np.atleast_2d(sigs)

    mask_is_list = isinstance(mask, list)
    if not mask_is_list:
        mask = [mask]

    comod_list = None
    if progress_bar:
        bar = ProgressBar(
            max_value=len(low_fq_range) * len(mask),
            title='comodulogram: %s' % model.get_title(name=True))
    for j, (filtered_low, filtered_high) in enumerate(extract(
            sigs=sigs, fs=fs, low_fq_range=low_fq_range,
            bandwidth=low_fq_width, fill=fill, ordar=ordar, enf=enf,
            random_noise=random_noise, normalize=normalize,
            whitening=whitening, draw='')):

        if high_sig is None:
            filtered_high = np.array(filtered_high)
            filtered_low = np.array(filtered_low)
        else:
            filtered_high = np.array(filtered_high[n_epochs:])
            filtered_low = np.array(filtered_low[:n_epochs])

        for i_mask, this_mask in enumerate(mask):
            sigdriv = filtered_low
            sigin = filtered_high
            sigin /= np.std(sigin)

            # fit the model DAR on the data
            model.fit(sigin=sigin, sigdriv=sigdriv, fs=fs, mask=this_mask)

            # get PSD difference
            spec, _ = model.amplitude_frequency()
            if method == 'minmax':
                spec_diff = spec.max(axis=1) - spec.min(axis=1)
            elif method == 'firstlast':
                spec_diff = spec[:, -1] - spec[:, 0]

            # crop the spectrum to high_fq_range
            frequencies = np.linspace(0, fs // 2, spec_diff.size)
            spec_diff = np.interp(high_fq_range, frequencies, spec_diff)

            # initialize the results arrays
            if comod_list is None:
                comod_list = []
                for _ in mask:
                    comod_list.append(np.zeros((low_fq_range.size,
                                                spec_diff.size)))

            # save the results
            comod_list[i_mask][j, :] = spec_diff

            if progress_bar:
                bar.update_with_increment_value(1)

    if not mask_is_list:
        return comod_list[0]
    else:
        return comod_list


def get_maximum_pac(comodulogram, low_fq_range, high_fq_range):
    """Get maximum PAC value in a comodulogram.
    'low_fq_range' and 'high_fq_range' must be the same than used in the
    modulation_index function that computed 'comodulogram'.

    Parameters
    ----------
    comodulogram  : PAC values, shape (len(low_fq_range), len(high_fq_range))
    low_fq_range  : low frequency range (phase signal)
    high_fq_range : high frequency range (amplitude signal)

    Return
    ------
    low_fq    : low frequency of maximum PAC
    high_fq   : high frequency of maximum PAC
    pac_value : maximum PAC value
    """
    i, j = argmax_2d(comodulogram)
    max_pac_value = comodulogram[i, j]

    low_fq = low_fq_range[i]
    high_fq = high_fq_range[j]

    return low_fq, high_fq, max_pac_value
