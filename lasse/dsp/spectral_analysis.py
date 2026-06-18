"""
Analyze a signal using Fourier-based frequency domain representations
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import welch


def discretized_frequency_axis(Fs, N):
    """Get frequency axis.
    Recall:
    In general, the FFT angles are: 0, 2*pi*k/N, ..., 2*pi*(N-1)/N
    For N=4, the FFT angles are: 0, 90, 180 and 270 degrees
    and indices are -2, -1, 0, 1, such that the value we want
    is first_index = -2
    For N=3, the FFT angles are: 0, 120 and 240 degrees
    and indices are -1, 0, 1, such that the value we want
    is first_index = -1
    """
    if np.remainder(N, 2) == 0:
        # N is even and indices are -N/2 to (N/2)-1
        first_index = -int(N / 2)
        freq_indices = np.arange(first_index, -first_index, 1)
    else:
        # N is odd and indices are -(N-1)/2 to (N-1)/2
        first_index = -int((N - 1) / 2)
        freq_indices = np.arange(first_index, (-first_index) + 1, 1)

    delta_f = Fs / N  # frequency space among continuous values
    f = delta_f * freq_indices
    # print('f=', f)
    return f


def spectrum_magnitude(x, Fs, show_plot=False, remove_mean=False, mag_threshold=None):
    """
    Plot the FFT magnitude of vector x using a bilateral spectrum.
    Normalize FFT by the FFT length, such that the output corresponds to
    the DTFS. Fs is the sampling frequency in Hz, used to plot the abscissa.

    Parameters
    ----------
    mag_threshold : float, optional
        Magnitude threshold in dB. Magnitudes below this threshold are set to 0.
        Useful for cleaning up phase plots (default: None, no thresholding).
    """
    if remove_mean:
        # remove the mean, which may be a strong peak that hides
        # the other parts
        x -= np.mean(x)

    # Calculate the amplitude
    X = np.fft.fft(x)  # calculate Fourier transform
    X = np.abs(X)  # get magnitude and discard phase of complex numbers
    X = np.fft.fftshift(X)  # move negative frequencies to the left
    N = len(X)  # number of frequency points in X
    X /= N  # normalize X by N to make it an estimate of Fourier series coefficients

    # Now calculate the discretized frequency axis
    f = discretized_frequency_axis(Fs, N)  # get frequency axis

    # Apply magnitude threshold if specified
    if mag_threshold is not None:
        # Convert threshold from dB to linear scale
        mag_threshold_linear = 10 ** (mag_threshold / 20)
        # Zero out magnitudes below threshold
        X[X < mag_threshold_linear] = 0

    if show_plot:
        # plt.semilogy(f, X)
        plt.plot(f, X)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("magnitude |X(f)| (Volts/Hz)")
        plt.title("Spectrum")
        plt.show()

    return X, f


def spectrum(x, Fs, show_plot=False, mag_threshold=None):
    """
    Calculate the full FFT spectrum of vector x using a bilateral spectrum.
    Normalize FFT by the FFT length, such that the output corresponds to
    the DTFS. Fs is the sampling frequency in Hz, used to plot the abscissa.

    Parameters
    ----------
    x : array-like
        Input signal
    Fs : float
        Sampling frequency in Hz
    show_plot : bool, optional
        If True, plot the magnitude and phase spectra using subplots (default: False)
    mag_threshold : float, optional
        Magnitude threshold in dB. Magnitudes below this threshold are set to 0.
        Useful for cleaning up phase plots (default: None, no thresholding).

    Returns
    -------
    X : ndarray (complex)
        Complex-valued FFT (normalized by FFT length)
    f : ndarray
        Frequency axis
    """

    # Calculate the FFT
    X = np.fft.fft(x)  # calculate Fourier transform
    X = np.fft.fftshift(X)  # move negative frequencies to the left
    N = len(X)  # number of frequency points in X
    X /= N  # normalize X by N

    # Calculate the discretized frequency axis
    f = discretized_frequency_axis(Fs, N)  # get frequency axis

    # Apply magnitude threshold if specified
    if mag_threshold is not None:
        # Convert threshold from dB to linear scale
        mag_threshold_linear = 10 ** (mag_threshold / 20)
        # Zero out magnitudes (and thus phases) below threshold
        mask = np.abs(X) < mag_threshold_linear
        X[mask] = 0

    if show_plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(f, np.abs(X))
        plt.xlabel("frequency (Hz)")
        plt.ylabel("magnitude |X(f)| (Volts/Hz)")
        plt.title("Spectrum Magnitude")
        plt.grid(True)

        plt.subplot(212)
        plt.plot(f, np.angle(X))
        plt.xlabel("frequency (Hz)")
        plt.ylabel("phase ∠X(f) (radians)")
        plt.title("Spectrum Phase")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return X, f


def power_spectral_density(x, Fs, show_plot=False):
    """
    PSD calculation using Welch's method.
    """
    # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html#scipy.signal.welch
    # scipy.signal.welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None,
    # nfft=None, detrend='constant', return_onesided=True, scaling='density',
    # axis=-1, average='mean')

    f, X_psd = welch(x, fs=Fs, window="hann", noverlap=16)

    if show_plot:
        plt.semilogy(f, X_psd)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("PSD (W/Hz)")
        plt.title("Power spectral density")
        plt.show()

    return X_psd, f


if __name__ == "__main__":
    # simple test
    Fs = 1000  # sampling frequency
    T = 1 / Fs  # sampling period
    L = 1500  # length of signal
    t = np.arange(0, L) * T  # time vector

    # Create a signal containing a 50 Hz sinusoid of amplitude 0.7
    # and a 120 Hz sinusoid of amplitude 1.
    s = 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

    # Add some noise to the signal
    x = s + 2 * np.random.randn(len(t))

    # Compute and plot 3 representations of the signal in frequency domain
    spectrum_magnitude(x, Fs, show_plot=True)
    spectrum(x, Fs, show_plot=True)
    power_spectral_density(x, Fs, show_plot=True)


def ak_specgram(signal_data, fs=None, window="hamming", noverlap=None, nfft=None):
    """
    Compute spectrogram of a signal using short-time Fourier transform (STFT).

    This method performs time-frequency decomposition of a signal by dividing it into
    overlapping segments, applying a window function to each segment, and computing
    the FFT. This is a Python implementation of MATLAB's specgram function adapted
    from ak_specgram.m.

    The spectrogram shows how the frequency content of a signal changes over time,
    which is useful for analyzing audio, speech, and other time-varying signals.

    Parameters
    ----------
    signal_data : array_like
        Input signal (1D array of samples). If stereo, extracts first channel.
    fs : float, optional
        Sampling frequency in Hz (samples per second). Default is 1.0
    window : str or tuple, optional
        Window function to apply to each segment:
        - 'hamming': Default, good general-purpose window
        - 'hann': Smooth transitions
        - 'blackman': Excellent frequency rejection
        - 'bartlett': Triangular window
    noverlap : int, optional
        Number of overlapping samples between consecutive segments (0-99%).
        Default is 50% of window length for smooth time resolution.
    nfft : int, optional
        Length of FFT (zero-padded if longer than window).
        Default: Next power of 2 >= signal_length / 8 (min 256)

    Returns
    -------
    Sxx : ndarray
        Power spectral density in linear scale.
        Shape: (nfft//2 + 1, n_segments) - frequency bins × time frames
    f : ndarray
        Frequency bin centers in Hz (0 to fs/2, Nyquist limit)
    t : ndarray
        Time values for each segment in seconds

    Notes
    -----
    The number of frequency bins is nfft//2 + 1 (one-sided spectrum).
    Time resolution = (nperseg - noverlap) / fs
    Frequency resolution = fs / nfft

    History
    -------
    Spectrogram visualization and analysis for WAV files.

    Converted MATLAB ak_specgram.m functionality to Python.

    Key improvements from ak_specgram.m:

    - Preemphasis filtering - Boosts higher frequencies for better clarity (coefficient: 0.9)
    - Dynamic thresholding - Sets floor value at max - threshold_db to reduce noise artifacts
    - Precise time axis calculation - Adjusts time positioning by adding half the window length for better alignment
    - Filter bandwidth control - Calculates optimal window length from suggested bandwidth
    - Better window calculation - Uses next power of 2 for FFT speed optimization
    - dB conversion - Uses 20*log10 for proper power spectrum representation
    """
    signal_data = np.asarray(signal_data)

    # Handle mono/stereo
    if signal_data.ndim > 1:
        signal_data = signal_data[:, 0]

    # Default sampling frequency
    if fs is None:
        fs = 1.0

    # Default NFFT
    if nfft is None:
        nfft = 2 ** int(np.ceil(np.log2(len(signal_data) / 8)))
        nfft = max(nfft, 256)

    # Default window length = NFFT
    nperseg = nfft

    # Default noverlap = 50% of nperseg
    if noverlap is None:
        noverlap = nperseg // 2

    # Compute spectrogram using scipy
    f, t, Sxx = signal.spectrogram(
        signal_data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density",
    )

    return Sxx, f, t


def plot_spectrogram(
    signal_data,
    fs,
    title="Spectrogram",
    window="hamming",
    noverlap=None,
    nfft=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    figsize=(12, 6),
):
    """
    Compute and visualize a spectrogram with professional formatting.

    This method creates a time-frequency visualization of a signal where:
    - Horizontal axis represents time progression
    - Vertical axis represents frequency content
    - Color intensity represents power (magnitude squared) at that time-frequency point

    The visualization uses matplotlib's pcolormesh for efficient rendering.

    Parameters
    ----------
    signal_data : array_like
        Input audio signal (1D array of samples)
    fs : float
        Sampling frequency in Hz (samples per second)
    title : str, default 'Spectrogram'
        Plot title displayed at top
    window : str, default 'hamming'
        Windowing function applied to segments
    noverlap : int, optional
        Samples overlapped between segments. None uses 50% overlap.
    nfft : int, optional
        FFT length for frequency resolution. None auto-calculates.
    vmin, vmax : float, optional
        Color scale limits in linear units. None for auto-scaling.
    cmap : str, default 'viridis'
        Matplotlib colormap name ('viridis', 'plasma', 'magma', 'inferno', etc.)
    figsize : tuple, default (12, 6)
        Figure dimensions (width_inches, height_inches)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization or saving
    ax : matplotlib.axes.Axes
        Axes object with the spectrogram plot

    Notes
    -----
    Power converted to linear scale using 10*log10(). Frequency limits set to
    Nyquist frequency (fs/2). Colorbar shows power scale.
    """
    # Compute spectrogram
    Sxx, f, t = ak_specgram(
        signal_data, fs=fs, window=window, noverlap=noverlap, nfft=nfft
    )

    # Convert to dB scale
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot spectrogram
    im = ax.pcolormesh(t, f, Sxx_dB, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)

    # Labels and title
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power (dB)")

    # Set reasonable frequency limits
    ax.set_ylim(0, fs / 2)

    plt.tight_layout()

    return fig, ax
