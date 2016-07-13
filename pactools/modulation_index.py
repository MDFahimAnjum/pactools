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
from .plot_comodulogram import plot_comodulogram_histogram
from .preprocess import extract


def multiple_band_pass(sig, fs, f_range, f_width, n_cycles=None, method=1):
    """
    Band-pass filter the signal at multiple frequencies
    """
    fixed_n_cycles = n_cycles
    sig = crop_for_fast_hilbert(sig)
    f_range = np.atleast_1d(f_range)

    if method == 1:
        fir = Carrier()

    n_frequencies = f_range.shape[0]
    res = np.zeros((n_frequencies, sig.shape[0]), dtype=np.complex128)
    for i, f in enumerate(f_range):
        # evaluate the number of cycle for this f_width and f
        if fixed_n_cycles is None:
            n_cycles = 1.65 * f / f_width

        # 0--------- with mne.filter.band_pass_filter
        if method == 0:
            low_sig = band_pass_filter(
                sig, Fs=fs, Fp1=f - f_width / 2.0, Fp2=f + f_width / 2.0,
                l_trans_bandwidth=f_width / 4.0,
                h_trans_bandwidth=f_width / 4.0,
                n_jobs=1, method='iir')

        # 1--------- with pactools.Carrier
        if method == 1:
            fir.design(fs, f, n_cycles, None, zero_mean=True)
            low_sig = fir.direct(sig)

        # common to the two methods
        res[i, :] = hilbert(low_sig)

    return res


def _modulation_index(filtered_low, filtered_high, method, fs, n_surrogates,
                      progress_bar, draw_phase):
    """
    Helper for modulation_index
    """
    rng = RandomState(42)

    tmax = filtered_low.shape[1]
    n_low = filtered_low.shape[0]
    n_high = filtered_high.shape[0]

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
    if progress_bar:
        bar = ProgressBar('comodulogram: %s' % method, max_value=n_low)
    MI = np.zeros((n_low, n_high))
    exp_phase = None
    for i in range(n_low):
        if method != 'tort':
            exp_phase = np.exp(1j * filtered_low[i])

        for j in range(n_high):
            MI[i, j] = _one_modulation_index(
                amplitude=filtered_high[j], phase=filtered_low[i],
                exp_phase=exp_phase, norm_a=norm_a[j], method=method,
                tmax=tmax, fs=fs, n_surrogates=n_surrogates,
                random_state=rng, draw_phase=draw_phase)

        if progress_bar:
            bar.update(i + 1)

    return MI


def _one_modulation_index(amplitude, phase, exp_phase, norm_a, method,
                          tmax, fs, n_surrogates, random_state, draw_phase):
    if method == 'ozkurt':
        # Modulation index as in Ozkurt 2011
        MI = np.abs(np.mean(amplitude * exp_phase))

        MI /= norm_a
        MI *= np.sqrt(tmax)

    elif method == 'canolty':
        # Modulation index as in Canolty 2006
        MI = np.abs(np.mean(amplitude * exp_phase))

        if n_surrogates > 0:
            # compute surrogates MIs
            MI_surr = np.empty(n_surrogates)
            for s in range(n_surrogates):
                shift = random_state.randint(fs, tmax - fs)
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


def modulation_index(fs, low_sig, high_sig=None,
                     low_fq_range=np.linspace(1.0, 10.0, 50),
                     low_fq_width=0.5,
                     high_fq_range=np.linspace(5.0, 150.0, 60),
                     high_fq_width=10.0,
                     method='tort',
                     n_surrogates=100,
                     draw=False, save_name=None,
                     vmin=None, vmax=None,
                     progress_bar=True,
                     draw_phase=False):
    """
    Compute the modulation index (MI) for Phase Amplitude Coupling (PAC).

    Parameters
    ----------
    fs            : sampling frequency
    low_sig       : one dimension signal where we extract the phase signal
    high_sig      : one dimension signal where we extract the amplitude signal
                    if None, we use low_sig for both signals
    low_fq_range  : low frequency range to compute the MI (phase signal)
    low_fq_width  : width of the band-pass filter
    high_fq_range : high frequency range to compute the MI (amplitude signal)
    high_fq_width : width of the band-pass filter
    method        : normalization method, in ('ozkurt', 'canolty', 'tort')
    n_surrogates  : number of surrogates computed in 'canolty's method
    draw          : if True, draw the comodulogram
    vmin, vmax    : if not None, it define the min/max value of the plot
    draw_phase    : if True, plot the phase distribution in 'tort' index

    Return
    ------
    MI            : Modulation Index,
                    shape (len(low_fq_range), len(high_fq_range))
    """
    # convert to numpy array
    low_fq_range = np.asarray(low_fq_range)
    high_fq_range = np.asarray(high_fq_range)

    if method in ('ozkurt', 'canolty', 'tort'):
        low_sig = low_sig.ravel()
        low_sig = crop_for_fast_hilbert(low_sig)
        if high_sig is not None:
            high_sig = high_sig.ravel()
            high_sig = crop_for_fast_hilbert(high_sig)
        else:
            high_sig = low_sig

        # compute a number of band-pass filtered and Hilbert filtered signals
        filtered_high = multiple_band_pass(high_sig, fs,
                                           high_fq_range, high_fq_width)
        filtered_low = multiple_band_pass(low_sig, fs,
                                          low_fq_range, low_fq_width)

        MI = _modulation_index(filtered_low, filtered_high, method, fs,
                               n_surrogates, progress_bar, draw_phase)
    elif isinstance(method, DAR):
        MI = driven_comodulogram(fs, low_sig, high_sig, model=method,
                                 low_fq_range=low_fq_range,
                                 low_fq_width=low_fq_width,
                                 high_fq_range=high_fq_range,
                                 progress_bar=progress_bar)
    else:
        raise(ValueError, 'unknown method: %s' % method)

    if draw:
        plot_comodulogram_histogram(MI, low_fq_range, low_fq_width,
                                    high_fq_range, high_fq_width,
                                    method, vmin, vmax, save_name)

    return MI


def driven_comodulogram(fs, low_sig, high_sig, model, low_fq_range,
                        low_fq_width, high_fq_range, method='minmax',
                        fill=4, ordar=12, enf=50., random_noise=None,
                        normalize=True, whitening=None,
                        progress_bar=True):
    """
    Compute the driven comodulogram with a DAR model

    sig            : single signal
    fs             : sampling frequency
    model          : DAR instance
    low_fq_range   : range of carrier frequency to scan
    methods        : 'firstlast' or 'minmax'
    """
    if high_sig is None:
        sigs = [low_sig.ravel()]
    else:
        sigs = [low_sig.ravel(), high_sig.ravel()]

    comod = None
    if progress_bar:
        bar = ProgressBar(
            max_value=len(low_fq_range) - 1,
            title='comodulogram: %s' % model.get_title(name=True))
    for j, (sigdrivs, sigins) in enumerate(extract(
            sigs=sigs, fs=fs, low_fq_range=low_fq_range,
            bandwidth=low_fq_width, fill=fill, ordar=ordar, enf=enf,
            random_noise=random_noise, normalize=normalize,
            whitening=whitening)):

        if high_sig is None:
            sigin = sigins[0]
            sigdriv = sigdrivs[0]
        else:
            sigin = sigins[1]
            sigdriv = sigdrivs[0]

        sigin /= np.std(sigin)
        model.fit(sigin=sigin, sigdriv=sigdriv, fs=fs)

        # get PSD difference
        spec, _ = model.amplitude_frequency()
        if method == 'minmax':
            spec_diff = spec.max(axis=1) - spec.min(axis=1)
        elif method == 'firstlast':
            spec_diff = spec[:, -1] - spec[:, 0]

        # crop the spectrum to high_fq_range
        frequencies = np.linspace(0, fs // 2, spec_diff.size)
        spec_diff = np.interp(high_fq_range, frequencies, spec_diff)

        # save in an array
        if comod is None:
            comod = np.zeros((low_fq_range.size, spec_diff.size))
        comod[j] = spec_diff

        if progress_bar:
            bar.update(j)

    return comod


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
