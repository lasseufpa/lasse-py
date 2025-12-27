"""
Analyze a signal in frequency domain.
"""

import matplotlib.pyplot as plt
import numpy as np
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


def spectrum_magnitude(x, Fs, show_plot=False, remove_mean=False):
    """
    Plot the FFT magnitude of vector x using a bilateral spectrum.
    Normalize FFT by the FFT length, such that the output corresponds to
    the DTFS. Fs is the sampling frequency in Hz, used to plot the abscissa.
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

    if show_plot:
        # plt.semilogy(f, X)
        plt.plot(f, X)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("magnitude |X(f)| (Volts/Hz)")
        plt.title("Spectrum")
        plt.show()

    return X, f


def spectrum(x, Fs, show_plot=False):
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

    # Compute and plot the spectrum
    print('ddd')
    spectrum_magnitude(x, Fs, show_plot=True)
    spectrum(x, Fs, show_plot=True)
    power_spectral_density(x, Fs, show_plot=True)
