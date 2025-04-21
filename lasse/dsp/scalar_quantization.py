"""
Methods to quantize and dequantize.
"""
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


def ak_quantizer(input, delta, b):
    """
    Applies a uniform mid-tread quantization to the input signal.

    The quantizer allocates:
        - 2^(b-1) levels to negative values,
        - 1 level to zero,
        - and 2^(b-1) - 1 levels to positive values.

    Values beyond the representable range are clipped to the nearest quantization level.

    Parameters:
        input (np.ndarray): Input signal to be quantized (can be any shape).
        delta (float): Quantization step size.
        b (int): Number of bits used for quantization.

    Returns:
        x_i (np.ndarray): Quantized integer indices, same shape as input.
        x_q (np.ndarray): Quantized signal (reconstructed), same shape as input.
    """
    x = input.ravel()  # create a one-dimensional view of the input array
    x_q = np.zeros(x.shape, dtype=float)
    x_i = np.zeros(x.shape, dtype=int)
    for i in range(len(x)):
        auxi = x[i] / delta  # Quantizer levels
        auxi = np.round(auxi)  # get the nearest integer
        if auxi > ((2 ** (b - 1)) - 1):
            auxi = (2 ** (b - 1)) - 1  # force a maximum value
        elif auxi < -(2 ** (b - 1)):
            auxi = -(2 ** (b - 1))  # force a minimum value
        auxq = auxi * delta  # get the decoded output already quantized
        x_q[i] = auxq
        x_i[i] = auxi
    x_q = x_q.reshape(input.shape)  # back to the dimension of input array
    x_i = x_i.reshape(input.shape)
    return x_i, x_q


def int_to_bitarray2_numpy_array(xi, num_of_bits):
    """
    Converts a 1D array of integers into cumulative binary masks of length 2**num_of_bits.

    This is not a bitwise representation, but rather a "soft histogram"-like encoding
    where each integer `k` is transformed into an array of `k` ones followed by zeros.

    Args:
        xi (np.ndarray): 1D array of non-negative integers, each less than 2**num_of_bits.
        num_of_bits (int): Determines the output vector size as 2**num_of_bits.

    Returns:
        np.ndarray: A 2D array of shape (len(xi), 2**num_of_bits), where each row contains
                    `xi[i]` ones followed by zeros.

    Example:
        xi = np.array([3, 5]) and num_of_bits = 3 (→ vector length = 8)
        → array([[1, 1, 1, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 0, 0, 0]], dtype=uint8)
    """
    N = len(xi)  # assume it's 1D
    num_levels = 2 ** num_of_bits
    out = np.zeros((N, num_levels), dtype=np.uint8)
    for i in range(N):
        out[i, 0 : xi[i]] = 1
    return out


def int_to_bitarray_numpy_array(xi, num_of_bits):
    """
    Converts a 1D array of integers to their binary representations.

    Args:
        xi (np.ndarray): 1D array of non-negative integers.
        num_of_bits (int): Number of bits to use in the binary representation.

    Returns:
        np.ndarray: A 2D array of shape (len(xi), num_of_bits), where each row is the
                    binary representation of the corresponding integer in `xi`.

    Example:
        xi = np.array([3, 5])
        → array([[0, 1, 1],
                 [1, 0, 1]], dtype=uint8)
    """
    N = len(xi)  # assume it's 1D
    out = np.zeros((N, num_of_bits), dtype=np.uint8)
    for i in range(N):
        out[i] = np.asarray(int_to_bitarray(xi[i], num_of_bits))
    return out


def int_to_bitarray(n, num_of_bits):
    """
    Converts an integer to a binary representation in the form of a NumPy array.

    Args:
        n (int): The integer to convert (should be non-negative and less than 2**num_of_bits).
        num_of_bits (int): The number of bits to use in the binary representation.

    Returns:
        np.ndarray: A 1D array of uint8 values (0 or 1), representing the binary encoding
                    of the integer `n` in big-endian order (most significant bit first).

    Example:
        int_to_bitarray(5, 4) → array([0, 1, 0, 1])
    """
    out = np.zeros((num_of_bits,), dtype=np.uint8)
    mask = 1
    for i in range(num_of_bits):
        out[num_of_bits - 1 - i] = mask & n
        n = n >> 1
    return out


def bitarray_to_int(bitarray):
    """
    Convert numpy array of bits 0 and 1 (not strings) into integer.

    Args:
        bitarray (np.ndarray): 1D array of bits (0 or 1).

    Returns:
        int: The integer value represented by the bit array.
    """
    num_of_bits = len(bitarray)
    out = np.uint64(0)
    for i in range(num_of_bits):
        factor = bitarray[num_of_bits - 1 - i] << i
        out += factor
    return out


class oldUniformQuantizer:
    """
    Legacy implementation of a uniform quantizer.

    Attributes:
        num_bits (int): Number of bits used for quantization.
        delta (float): Step size between quantization levels.
        quantizerLevels (np.ndarray): Array of quantization levels.
        xminq (float): Minimum quantizer output level.
        xi_max_index (int): Maximum index for quantization.
    """

    def __init__(self, num_bits, xmin, xmax, forceZeroLevel=False):
        self.num_bits = num_bits
        M = 2 ** num_bits  # number of quantization levels

        # Choose the min value such that the result coincides with Matlab Lloyd's
        # optimum quantizer when the input is uniformly distributed. Instead of
        # delta=abs((xmax-xmin)/(M-1)) #as quantization step use:
        self.delta = abs((xmax - xmin) / M)  # quantization step
        self.quantizerLevels = (
            xmin + (self.delta / 2.0) + np.arange(M) * self.delta
        )  # output values
        if forceZeroLevel:
            # np.nonzero plays the role of Matlab's find
            isZeroRepresented = np.nonzero(self.quantizerLevels == 0)  # is 0 there?
            if isZeroRepresented[0].size == 0:  # zero is not represented yet
                min_abs = np.min(np.abs(self.quantizerLevels))
                minLevelIndices = np.nonzero(np.abs(self.quantizerLevels) == min_abs)[0]
                closestInd = minLevelIndices[-1]  # end
                closestToZeroValue = self.quantizerLevels[closestInd]
                self.quantizerLevels = self.quantizerLevels - closestToZeroValue

        self.xminq = np.min(self.quantizerLevels)  # keep to speed up quantize()
        self.xi_max_index = (2 ** self.num_bits) - 1


class UniformQuantizer:
    """
    Uniform quantizer with optional zero-centered level (mid-tread design).

    Attributes:
        num_bits (int): Number of bits used in quantization.
        delta (float): Step size between quantization levels.
        quantizerLevels (np.ndarray): The midpoints of quantization bins.
        xminq (float): Minimum quantizer output level.
        xi_max_index (int): Maximum index for quantization.
    """

    def __init__(self, num_bits, xmin, xmax, forceZeroLevel=False):
        """
        Initialize the UniformQuantizer.

        Parameters:
            num_bits (int): Number of quantization bits.
            xmin (float): Minimum representable input value.
            xmax (float): Maximum representable input value.
            forceZeroLevel (bool): Whether to ensure 0 is exactly one of the quantizer levels.
                                   If True, more levels will be allocated to the negative side.
        """
        self.num_bits = num_bits
        M = 2 ** num_bits  # total number of quantization levels
        self.delta = abs((xmax - xmin) / M)

        if forceZeroLevel:
            half = M // 2
            if M % 2 == 0:
                indices = np.arange(-half, half)
            else:
                indices = np.arange(-half, half + 1)
            self.quantizerLevels = indices * self.delta
        else:
            self.quantizerLevels = xmin + (self.delta / 2.0) + np.arange(M) * self.delta

        self.xminq = np.min(self.quantizerLevels)
        self.xi_max_index = M - 1

    def __repr__(self):
        return (
            f"UniformQuantizer(num_bits={self.num_bits}, delta={self.delta:.4f}, "
            f"forceZeroLevel={'True' if 0 in self.quantizerLevels else 'False'}, "
            f"quantizerLevels={self.quantizerLevels})"
        )

    def get_quantizer_levels(self):
        """
        Get the quantization levels.

        Returns:
            np.ndarray: Array of quantization levels.
        """
        return self.quantizerLevels

    def get_partition_thresholds(self):
        """
        Get the partition thresholds between quantization levels.

        Returns:
            np.ndarray: Array of partition thresholds.
        """
        partitionThresholds = 0.5 * (
            self.quantizerLevels[0:-2] + self.quantizerLevels[1:-1]
        )
        return partitionThresholds

    def quantize_numpy_array(self, x):
        """
        Quantize a NumPy array.

        Args:
            x (np.ndarray): Input array to quantize.

        Returns:
            tuple: Quantized array and corresponding indices.
        """
        x_i = np.array((x - self.xminq) / self.delta)
        x_i = np.round(x_i)
        x_i[x_i < 0] = 0
        x_i[x_i > self.xi_max_index] = self.xi_max_index
        x_q = x_i * self.delta + self.xminq
        return x_q, x_i.astype(np.int64)

    def dequantize_numpy_array(self, x_i):
        """
        Dequantize a NumPy array of indices.

        Args:
            x_i (np.ndarray): Array of quantized indices.

        Returns:
            np.ndarray: Dequantized array.
        """
        x_q = x_i.astype(np.float32) * self.delta + self.xminq
        return x_q

    def quantize_real_scalar(self, x):
        """
        Quantize a single scalar value.

        Args:
            x (float): Scalar value to quantize.

        Returns:
            tuple: Quantized value and corresponding index.
        """
        x_i = (x - self.xminq) / self.delta
        x_i = np.round(x_i)
        if x_i < 0:
            x_i = 0
        if x_i > self.xi_max_index:
            x_i = self.xi_max_index
        x_q = x_i * self.delta + self.xminq
        return x_q, int(x_i)


class OneBitUniformQuantizer(UniformQuantizer):
    """
    Specialized uniform quantizer for 1-bit quantization.

    Attributes:
        threshold (float): Threshold value for quantization.
    """

    def __init__(self, threshold=0):
        """
        Initialize the OneBitUniformQuantizer.

        Parameters:
            threshold (float): Threshold value for quantization.
        """
        num_bits = 1
        xmin = np.finfo(np.float64).min
        xmax = np.finfo(np.float64).max
        UniformQuantizer.__init__(self, num_bits, xmin, xmax)
        self.threshold = threshold

    def quantize(self, x):
        """
        Perform 1-bit quantization.

        Args:
            x (np.ndarray): Input array to quantize.

        Returns:
            np.ndarray: Boolean array indicating whether values exceed the threshold.
        """
        return x > self.threshold


if __name__ == "__main__":
    num_bits = 3
    xmin = -2
    xmax = 2
    forceZeroLevel = True
    uniformQuantizer = UniformQuantizer(
        num_bits, xmin, xmax, forceZeroLevel=forceZeroLevel
    )
    print(uniformQuantizer.__repr__())

    # quantize scalar
    x = -0.4
    xq, xi = uniformQuantizer.quantize_real_scalar(x)
    xq2 = uniformQuantizer.dequantize_numpy_array(np.array(xi))
    print(xq, xi, xq2)

    # convert to bits
    num_bits = 8
    print(int_to_bitarray(xi, num_bits))

    # quantize array
    x = -1.4 * np.ones((3, 2))
    xq, xi = uniformQuantizer.quantize_numpy_array(x)
    print(xq, xi)
    # prepare to convert to bit
    y = xi.flatten().astype(np.int64)
    z = int_to_bitarray_numpy_array(y, num_bits)
    print(z)
    num_bits = 4
    z2 = int_to_bitarray2_numpy_array(y, num_bits)
    print(z2)
