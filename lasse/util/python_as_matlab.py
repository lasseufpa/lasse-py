'''
Python code that mimics Matlab / Octave

Python functions/methods do not know about how many outputs are requested:
https://stackoverflow.com/questions/35389437/do-python-functions-know-how-many-outputs-are-requested
Hence, to comply with Matlab, we will create all outputs.
'''

import numpy as np
from scipy import signal


def xcorr(x, y=None, max_lag=-1, scaling_option='none'):
    '''
    Cross-correlation.

    scaling_option — Normalization option. Same as Matlab:
    'none' (default) | 'biased' | 'unbiased' | 'normalized' | 'coeff'

    Python has a numpy implementation np.correlate but we will use scipy:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html#scipy.signal.correlate
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlation_lags.html#scipy.signal.correlation_lags
    Aldebaro. Jan 2024.
    '''
    if y is None:
        y = x  # autocorrelation

    # calculate cross-correlation
    Rxy = signal.correlate(x, y, mode='full')
    # and its lags
    lags = signal.correlation_lags(x.size, y.size, mode="full")

    if max_lag != -1:  # in case max_lag is specified
        zero_lag_index = np.where(lags == 0)[0][0]
        first_lag = zero_lag_index - max_lag
        last_lag = zero_lag_index + max_lag
        Rxy = Rxy[first_lag:last_lag + 1]
        lags = lags[first_lag:last_lag + 1]

    if scaling_option == 'none':
        return Rxy, lags
    else:
        if x.size != y.size:
            raise Exception("Scaling must be \'none\' if vectors have distinct sizes!")
        N = x.size  # which coincides with y.size
        if scaling_option == 'biased':
            # Normalize by dividing by the number of samples of longest vector
            return Rxy / N, lags
        elif scaling_option == 'unbiased':
            denominator = N - np.abs(lags)
            return Rxy / denominator, lags
        elif scaling_option == 'normalized' or scaling_option == 'coeff':
            sum_squared_x = np.sum(np.abs(x) ** 2)
            sum_squared_y = np.sum(np.abs(y) ** 2)
            return Rxy / np.sqrt(sum_squared_x * sum_squared_y), lags


def test_xcorr():
    '''
    % Compare the results with the ones from Matlab / Octave below:
    y = [4 + 1j, 2 - 3j, 1, -5, 2];
    x = [1 + 1j, 2 - 1j, 3];
    Rxy = xcorr(x);
    disp(num2str(Rxy))
    [Rxy, lags]= xcorr(x, conj(x), 'none');
    disp(num2str(Rxy))
    [Rxy, lags]= xcorr(x, conj(x), 1);
    disp(num2str(Rxy))
    y = [2 - 3j, -5, 2];
    [Rxy, lags]= xcorr(x, y, 1, 'biased');
    disp(num2str(Rxy))
    [Rxy, lags]= xcorr(x, y, 1, 'unbiased');
    disp(num2str(Rxy))
    [Rxy, lags]= xcorr(x, y, 'coeff');
    disp(num2str(Rxy))
    disp(num2str(lags))

    '''
    y = np.array([4 + 1j, 2 - 3j, 1, -5, 2])
    x = np.array([1 + 1j, 2 - 1j, 3])
    Rxy = xcorr(x)
    print(Rxy)
    Rxy, lags = xcorr(x, np.conj(x), scaling_option='none')
    print(Rxy)
    Rxy, lags = xcorr(x, np.conj(x), max_lag=1)
    print(Rxy)
    y = np.array([2 - 3j, -5, 2])
    Rxy, lags = xcorr(x, y=y, max_lag=1, scaling_option='biased')
    print(Rxy)
    Rxy, lags = xcorr(x, y=y, max_lag=1, scaling_option='unbiased')
    print(Rxy)
    Rxy, lags = xcorr(x, y=y, scaling_option='coeff')
    print(Rxy)
    print(lags)


if __name__ == "__main__":
    test_xcorr()
