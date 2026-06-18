"""
Methods to design mid-tread and mid-rise quantizers,
and also quantize and dequantize a signal.
"""
import numpy as np


class BitUtils:
    @staticmethod
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

    @staticmethod
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
            out[i] = np.asarray(BitUtils.int_to_bitarray(xi[i], num_of_bits))
        return out

    @staticmethod
    def int_to_bitarray(n, num_of_bits):
        """
        Converts an integer to a binary representation in the form of a NumPy array.

        Args:
            n (int): The integer to convert.
            num_of_bits (int): Number of bits in the representation.

        Returns:
            np.ndarray: Binary representation (big-endian).

        Raises:
            ValueError: If n is negative or cannot be represented with num_of_bits.
            TypeError: If inputs are not integers.
        """

        # --- Type checking ---
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"n must be an integer, got {type(n)}")

        if not isinstance(num_of_bits, int):
            raise TypeError(f"num_of_bits must be an integer, got {type(num_of_bits)}")

        if num_of_bits <= 0:
            raise ValueError("num_of_bits must be positive")

        # --- Range checking ---
        max_val = (1 << num_of_bits) - 1

        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")

        if n > max_val:
            raise ValueError(
                f"n={n} cannot be represented with {num_of_bits} bits "
                f"(max={max_val})"
            )

        # --- Conversion ---
        out = np.zeros((num_of_bits,), dtype=np.uint8)

        for i in range(num_of_bits):
            out[num_of_bits - 1 - i] = (n >> i) & 1

        return out

    @staticmethod
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


class Quantizer:
    def __init__(self, num_bits):
        """
        Initialize the Quantizer object.

        Args:
            num_bits (int): Number of bits for quantization.

        Raises:
            TypeError: If num_bits is not an integer.
            ValueError: If num_bits is not positive.

        Attributes:
            num_bits (int): Number of bits for quantization.
            M (int): Equivalent to 2 ** num_bits.
            xi_max (int): Maximum index (from 0 to this maximum)
        """
        if not isinstance(num_bits, int):
            raise TypeError(f"num_bits must be an integer, got {type(num_bits)}")
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")
        self.num_bits = num_bits
        self.M = 1 << num_bits  # equivalent to 2 ** num_bits
        self.xi_max = self.M - 1  # maximum index (from 0 to this maximum)

    def quantize(self, x):
        """
        Quantize a given input array.

        Args:
            x (np.ndarray): Input array to be quantized.

        Returns:
            np.ndarray: Quantized array.
        """
        x = self._validate_input(x)
        return self._quantize_implementation(x)

    def dequantize(self, xi):
        """
        Dequantize a given index array.

        Args:
            xi (np.ndarray): Index array to be dequantized.

        Returns:
            np.ndarray: Dequantized array.
        """
        xi = np.asarray(xi)
        return self._dequantize_implementation(xi)

    def _validate_input(self, x):
        """
        Validate the input array for quantization/dequantization.

        Args:
            x (np.ndarray): Input array to be validated.

        Returns:
            np.ndarray: Validated input array with float64 dtype.
        """
        return np.asarray(x, dtype=np.float64)

    def _quantize_implementation(self, x):
        """
        This method should be implemented by subclasses to perform the actual quantization.
        It should take a float64 numpy array as input and return a float64 numpy array as output.
        The output array should have the same shape as the input array.
        The output values should be the quantized values of the input array.
        """
        raise NotImplementedError("Subclasses must implement this")

    def _dequantize_implementation(self, xi):
        """
        This method should be implemented by subclasses to perform the actual dequantization.
        It should take an integer numpy array as input and return a float64 numpy array as output.
        The output array should have the same shape as the input array.
        The output values should be the dequantized values of the input array.
        """
        raise NotImplementedError("Subclasses must implement this")


class UniformQuantizer(Quantizer):
    """
    Uniform quantizer with optional zero-centered level (mid-tread design).

    @TODO: Choose the min value such that the result coincides with Matlab Lloyd's

    Attributes:
        num_bits (int): Number of bits used in quantization.
        delta (float): Step size between quantization levels.
        quantizerLevels (np.ndarray): The midpoints of quantization bins.
        xminq (float): Minimum quantizer output level.
        xi_max_index (int): Maximum index for quantization.
    """

    def __init__(self, num_bits: int, delta: float, xminq: float):
        """
        Initialize the UniformQuantizer.

        Parameters:
            num_bits (int): Number of bits used in quantization.
            delta (float): Step size between quantization levels.
            xminq (float): Minimum quantizer output level.
        """
        super().__init__(num_bits)
        self.delta = delta
        self.quantizerLevels = xminq + np.arange(2 ** num_bits) * delta
        self.xminq = xminq

    @staticmethod
    def midtread_uniform_symmetric_quantize(input, delta, b):

        x = np.asarray(input, dtype=np.float64)
        original_shape = x.shape

        offset = 2 ** (b - 1)

        xi_signed = np.round(x / delta)
        xi_signed = np.clip(xi_signed, -offset, offset - 1)

        xi = (xi_signed + offset).astype(np.int64)  # FIX
        xq = xi_signed * delta

        if x.ndim == 0:
            return int(xi), float(xq)

        return xi.reshape(original_shape), xq.reshape(original_shape)

    @staticmethod
    def design_midtread_asymmetric_auto(num_bits, xmin, xmax):
        """
        Mid-tread quantizer with automatic asymmetric level allocation.

        Zero is guaranteed to be a level, and the number of negative levels
        is chosen proportionally to the signal dynamic range.
        """

        if xmin >= 0 or xmax <= 0:
            raise ValueError("Range must include zero for mid-tread design")

        M = 2 ** num_bits

        # --- Step 1: allocate levels proportionally ---
        neg_levels = int(round(M * abs(xmin) / (abs(xmin) + xmax)))

        # clamp to valid range
        neg_levels = max(1, min(M - 1, neg_levels))

        # --- Step 2: define index range ---
        k_min = -neg_levels
        k_max = M - neg_levels - 1

        # --- Step 3: compute Δ to cover full range ---
        delta_candidates = []

        if neg_levels > 0:
            delta_candidates.append(abs(xmin) / neg_levels)

        if k_max > 0:
            delta_candidates.append(xmax / k_max)

        delta = max(delta_candidates)

        # --- Step 4: build levels ---
        indices = np.arange(k_min, k_max + 1)
        levels = indices * delta

        return delta, np.min(levels)

    @staticmethod
    def design_quantizer(num_bits, xmin, xmax, force_midtread=False):
        """
        Design a uniform quantizer.

        Parameters:
            num_bits (int): Number of quantization bits.
            xmin (float): Minimum representable input value.
            xmax (float): Maximum representable input value.
            force_midtread (bool): Whether to ensure 0 is exactly one of the quantizer levels.
                                   If True, more levels will be allocated to the negative side.
        """
        M = 2 ** num_bits  # total number of quantization levels
        if xmax < xmin:
            raise Exception("xmax must be greater than xmin")

        # Check whether it makes sense to force a zero level
        if force_midtread:
            if xmin > 0 or xmax < 0:
                raise Exception(
                    "Cannot force zero level if range [xmin,  xmax] does not include zero."
                )

        if force_midtread:
            # Mid-tread quantizer (zero level included)
            delta, xminq = UniformQuantizer.design_midtread_asymmetric_auto(
                num_bits, xmin, xmax
            )
            # quantizerLevels = indices * delta
        else:
            # Mid-rise quantizer (no zero level)
            delta = (xmax - xmin) / float(M)
            quantizerLevels = xmin + (delta / 2.0) + np.arange(M) * delta
            # minimum quantizer output, which may be different from input xmin
            xminq = np.min(quantizerLevels)
        return delta, xminq

    def __repr__(self):
        """
        Return a string representation of the UniformQuantizer object.

        The string includes the number of quantization bits, the step size delta,
        whether the quantizer is forced to include zero (mid-tread design),
        and the quantization levels.

        Returns:
            str: String representation of the UniformQuantizer object.
        """
        return (
            f"UniformQuantizer(num_bits={self.num_bits}, delta={self.delta:.4f}, "
            f"force_midtread={'True' if 0 in self.quantizerLevels else 'False'}, "
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

    def _quantize_implementation(self, x):
        """
        Quantize a NumPy array.

        Args:
            x (np.ndarray): Input array to quantize.

        Returns:
            tuple: Quantized array and corresponding indices.
        """
        xi = np.round((x - self.xminq) / self.delta)
        xi = np.clip(xi, 0, self.xi_max).astype(np.int64)
        xq = xi * self.delta + self.xminq
        return xq, xi

    def _dequantize_implementation(self, xi):
        """
        Dequantize a NumPy array of indices.

        Args:
            xi (np.ndarray): Array of quantized indices.

        Returns:
            np.ndarray: Dequantized array.
        """
        return xi * self.delta + self.xminq

    def get_parameters(self) -> tuple[float, float, int]:
        # Return the parameters of the quantizer
        """
        Return the parameters of the quantizer.

        Returns:
            tuple: A tuple containing the minimum quantizer output level (xminq),
                the step size (delta), and the number of quantization bits (num_bits).
        """
        return self.xminq, self.delta, self.num_bits


class OneBitUniformQuantizer(Quantizer):
    """
    1-bit quantizer with configurable threshold and reconstruction levels.
    """

    def __init__(self, threshold=0.0, low_level=-1.0, high_level=1.0):
        super().__init__(num_bits=1)
        self.threshold = float(threshold)
        self.levels = np.array([low_level, high_level], dtype=np.float64)

    def _quantize_implementation(self, x):
        """
        Returns:
            x_q: reconstructed values
            xi: indices (0 or 1)
        """
        xi = (x > self.threshold).astype(np.int64)
        x_q = self.levels[xi]
        return x_q, xi

    def _dequantize_implementation(self, xi):
        """
        Map indices back to reconstruction levels.
        """
        xi = np.asarray(xi, dtype=np.int64)

        if np.any((xi < 0) | (xi > 1)):
            raise ValueError("Indices must be 0 or 1 for 1-bit quantizer")

        return self.levels[xi]

    def __repr__(self):
        return (
            f"OneBitUniformQuantizer(threshold={self.threshold}, "
            f"levels={self.levels})"
        )


class NonUniformQuantizer(Quantizer):
    """
    Non-uniform quantizer designed via Lloyd-Max algorithm.

    This implementation closely follows MATLAB's lloyds behavior:
    - Iterative optimization of decision thresholds and centroids
    - MSE distortion minimization
    - Sample-based design (empirical PDF)

    Attributes:
        num_bits (int)
        M (int): number of levels
        quantizerLevels (np.ndarray): reconstruction levels
        partitionThresholds (np.ndarray): decision boundaries
    """

    def __init__(self, num_bits, levels, thresholds):
        """
        Initialize NonUniformQuantizer.

        Parameters:
            num_bits (int): Number of quantization bits.
            levels (np.ndarray): Reconstruction levels.
            thresholds (np.ndarray): Decision boundaries.
        """
        super().__init__(num_bits)
        self.quantizerLevels = levels
        self.partitionThresholds = thresholds

    # ==========================================================
    # DESIGN (LLOYD-MAX)
    # ==========================================================
    @staticmethod
    def design_lloyd(x, num_bits, max_iter=100, tol=1e-7, init="uniform"):
        """
        Design a non-uniform quantizer using Lloyd-Max algorithm.

        Parameters:
            x (np.ndarray): training data (1D)
            num_bits (int): number of bits
            max_iter (int): max iterations
            tol (float): convergence tolerance (distortion change)
            init (str): initialization ('uniform' or 'random')

        Returns:
            levels, thresholds
        """
        x = np.asarray(x).ravel()
        M = 2 ** num_bits

        xmin, xmax = np.min(x), np.max(x)

        # ------------------------------------------
        # Initialization (important for MATLAB match)
        # ------------------------------------------
        if init == "uniform":
            levels = np.linspace(xmin, xmax, M)
        elif init == "random":
            levels = np.sort(np.random.choice(x, M, replace=False))
        else:
            raise ValueError("init must be 'uniform' or 'random'")

        thresholds = np.zeros(M - 1)

        prev_distortion = np.inf

        for it in range(max_iter):

            # ------------------------------------------
            # Step 1: update thresholds (midpoints)
            # ------------------------------------------
            thresholds = 0.5 * (levels[:-1] + levels[1:])

            # ------------------------------------------
            # Step 2: assign samples to regions
            # ------------------------------------------
            indices = np.digitize(x, thresholds)

            # ------------------------------------------
            # Step 3: update centroids
            # ------------------------------------------
            new_levels = np.copy(levels)

            for k in range(M):
                region = x[indices == k]
                if len(region) > 0:
                    new_levels[k] = np.mean(region)
                # else: keep previous level (MATLAB behavior)

            levels = new_levels

            # ------------------------------------------
            # Step 4: compute distortion
            # ------------------------------------------
            xq = levels[indices]
            distortion = np.mean((x - xq) ** 2)

            # convergence check
            if np.abs(prev_distortion - distortion) < tol:
                break

            prev_distortion = distortion

        return levels, thresholds

    @staticmethod
    def design_quantizer(x, num_bits, **kwargs):
        """
        Wrapper for Lloyd design.

        Returns:
            levels, thresholds
        """
        return NonUniformQuantizer.design_lloyd(x, num_bits, **kwargs)

    # ==========================================================
    # API (similar to UniformQuantizer)
    # ==========================================================
    def __repr__(self):
        """
        Return a string representation of the NonUniformQuantizer object.

        This string includes the number of quantization bits, the quantizer levels, and the partition thresholds.
        """
        return (
            f"NonUniformQuantizer(num_bits={self.num_bits}, "
            f"levels={self.quantizerLevels}, "
            f"thresholds={self.partitionThresholds})"
        )

    def get_quantizer_levels(self):
        """
        Get the quantizer levels.

        Returns:
            np.ndarray: the quantizer levels
        """
        return self.quantizerLevels

    def get_partition_thresholds(self):
        """
        Get the partition thresholds.

        Returns:
            np.ndarray: the partition thresholds
        """
        return self.partitionThresholds

    # ==========================================================
    # QUANTIZATION
    # ==========================================================
    def _quantize_implementation(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Quantize input data using the non-uniform quantizer.

        Parameters:
            x (np.ndarray): Input data to be quantized.

        Returns:
            tuple: A tuple containing the quantized data and the corresponding
                indices of the quantizer levels.
        """

        x = np.asarray(x)

        indices = np.digitize(x, self.partitionThresholds)
        x_q = self.quantizerLevels[indices]

        return x_q, indices.astype(np.int64)

    def _dequantize_implementation(self, xi):
        """
        Dequantize input data using the non-uniform quantizer.

        Parameters:
            x_i (np.ndarray): Input data to be dequantized.

        Returns:
            np.ndarray: the dequantized data
        """
        xi = np.asarray(xi)
        return self.quantizerLevels[xi]

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Get the parameters of the NonUniformQuantizer.

        Parameters:
            self (NonUniformQuantizer): The NonUniformQuantizer object.

        Returns:
            tuple: A tuple containing the quantizer levels (np.ndarray), partition thresholds (np.ndarray), and the number of quantization bits (int).
        """
        return self.quantizerLevels, self.partitionThresholds, self.num_bits


def test_conversion_to_bits():
    xi = np.array(np.arange(8))
    num_bits = 3
    # prepare to convert to bit
    y = xi.flatten().astype(np.int64)
    z = BitUtils.int_to_bitarray_numpy_array(y, num_bits)
    print("Conventional binary representation with ", num_bits, "bits:\n", z)
    z2 = BitUtils.int_to_bitarray2_numpy_array(y, num_bits + 1)
    print(
        "Alternative representation where the 1's indicate the number, using ",
        num_bits + 1,
        "bits:\n",
        z2,
    )


def test_uniform_quantizer():
    # First example of uniform quantizer (simpler one)
    num_bits = 2
    xmin = -8
    xmax = 1.3
    force_midtread = True

    delta, xminq = UniformQuantizer.design_quantizer(
        num_bits, xmin, xmax, force_midtread=force_midtread
    )

    uniformQuantizer = UniformQuantizer(num_bits, float(delta), float(xminq))
    print(uniformQuantizer.__repr__())

    # quantize scalar
    x = -0.4
    xq, xi = uniformQuantizer.quantize(x)
    xq2 = uniformQuantizer.dequantize(np.array(xi))
    print(xq, xi, xq2)

    # use static method
    delta = 0.2
    num_bits = 12
    xi, xq = UniformQuantizer.midtread_uniform_symmetric_quantize(x, delta, num_bits)
    print("mid-tread, symmetric quantization:\n", x, xi, xq)

    # Second example of uniform quantizer: Gaussian source
    x = np.random.randn(10000)
    xmin = np.mean(x) - 3.0 * np.std(x)  # bad choice: xmin = np.min(x)
    xmax = np.mean(x) + 3.0 * np.std(x)  # bad choice: xmax = np.max(x)
    num_bits = 3  # to compare with non-uniform quantizer
    delta, xminq = UniformQuantizer.design_quantizer(
        num_bits, xmin, xmax, force_midtread=True
    )

    uniformQuantizer = UniformQuantizer(num_bits, float(delta), float(xminq))
    print("Uniform quantizer:\n", uniformQuantizer)
    # quantize array,
    xq, xi = uniformQuantizer.quantize(x)
    print(xq, xi)
    print("MSE (uniform quantizer):", np.mean((x - xq) ** 2))

    xminq, delta, num_bits = uniformQuantizer.get_parameters()
    print(xminq, delta, num_bits)


def test_non_uniform_quantizer():
    np.random.seed(0)

    # Example: Gaussian source
    x = np.random.randn(10000)

    num_bits = 3

    quantization_levels, thresholds = NonUniformQuantizer.design_quantizer(
        x, num_bits, max_iter=200, tol=1e-9
    )

    q = NonUniformQuantizer(num_bits, quantization_levels, thresholds)

    print("Non-uniform quantizer:\n", q)

    # quantize scalar
    x0 = 0.3
    xq, idx = q.quantize(x0)
    print(xq, idx)

    # quantize array
    xq_arr, idx_arr = q.quantize(x)
    print("MSE: non-uniform quantizer", np.mean((x - xq_arr) ** 2))

    quantizerLevels, partitionThresholds, num_bits = q.get_parameters()
    print(quantizerLevels, partitionThresholds, num_bits)


def test_one_bit_quantizer():
    """
    Test the OneBitUniformQuantizer class.

    The test quantizes a small array, and then dequantizes the result.
    """

    q = OneBitUniformQuantizer(threshold=0)

    x = np.array([-2, -0.1, 0.2, 3])

    xq, xi = q.quantize(x)

    # xi = [0, 0, 1, 1]
    # xq = [-1, -1, 1, 1]

    xq2 = q.dequantize(xi)
    print(xq, xi, xq2)

    assert np.allclose(xq, xq2)


if __name__ == "__main__":
    print("test_conversion_to_bits()")
    test_conversion_to_bits()
    print("\n\ntest_uniform_quantizer()")
    test_uniform_quantizer()
    print("\n\ntest_non_uniform_quantizer()")
    test_non_uniform_quantizer()
    print("\n\ntest_one_bit_quantizer()")
    test_one_bit_quantizer()
