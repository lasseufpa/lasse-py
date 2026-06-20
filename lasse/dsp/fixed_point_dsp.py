"""
This module implements fixed-point IIR filtering and quantization of filter coefficients.
It includes functions to:
- Quantize filter coefficients and input signal to fixed-point representation.
- Apply the fixed-point IIR filter to an input signal.
- Compare the frequency responses of unquantized and quantized filters.
"""

import math
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import butter, freqz, lfilter


def saturate_int16(x: float | int | np.integer | np.floating) -> np.int16:
    """
    Clamp a numeric value to the signed int16 range and return int16.
    """
    return np.int16(np.clip(x, -32768, 32767))


def saturate_int8(x: float | int | np.integer | np.floating) -> np.int8:
    """
    Clamp a numeric value to the signed int8 range and return int8."""
    return np.int8(np.clip(x, -128, 127))


def int8_fixed_to_float(x_i: npt.ArrayLike, frac_bits: int) -> npt.NDArray[np.float64]:
    """
    Convert signed int8 fixed-point representation back to float.
    """
    return np.asarray(x_i, dtype=np.float64) / (2 ** frac_bits)


def compare_frequency_responses(
    B: npt.ArrayLike,
    A: npt.ArrayLike,
    Bq: npt.ArrayLike,
    Aq: npt.ArrayLike,
    worN: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare frequency responses of unquantized and quantized IIR filters.

    B, A   : unquantized numerator and denominator
    Bq, Aq : quantized numerator and denominator
    worN   : number of frequency samples
    """

    b_arr = np.asarray(B, dtype=np.float64)
    a_arr = np.asarray(A, dtype=np.float64)
    bq_arr = np.asarray(Bq, dtype=np.float64)
    aq_arr = np.asarray(Aq, dtype=np.float64)

    w, H = freqz(cast(Any, b_arr), cast(Any, a_arr), worN=worN)
    _, Hq = freqz(cast(Any, bq_arr), cast(Any, aq_arr), worN=worN)

    f = w / np.pi  # normalized frequency, where 1 corresponds to Nyquist

    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    magq_db = 20 * np.log10(np.maximum(np.abs(Hq), 1e-12))

    phase = np.unwrap(np.angle(H))
    phaseq = np.unwrap(np.angle(Hq))

    plt.figure()
    plt.plot(f, mag_db, label="Unquantized")
    plt.plot(f, magq_db, "--", label="Quantized")
    plt.xlabel(r"Normalized frequency $\omega/\pi$")
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude response")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(f, phase, label="Unquantized")
    plt.plot(f, phaseq, "--", label="Quantized")
    plt.xlabel(r"Normalized frequency $\omega/\pi$")
    plt.ylabel("Phase (rad)")
    plt.title("Phase response")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(f, magq_db - mag_db)
    plt.xlabel(r"Normalized frequency $\omega/\pi$")
    plt.ylabel("Magnitude error (dB)")
    plt.title("Quantization error in magnitude response")
    plt.grid(True)

    plt.show()

    return w, H, Hq


def apply_fixed_point_filter(
    B: npt.ArrayLike,
    A: npt.ArrayLike,
    x: npt.ArrayLike,
    total_bits: int = 8,
    frac_bits: int = 3,
    acc_bits: int = 16,
    acc_frac_bits: int = 6,
    mode: Literal["floor", "round", "trunc"] = "round",
) -> np.ndarray:
    """
    Apply fixed-point filtering to input signal x using quantized
    coefficients B and A. All these variables and intermediate results are quantized to fixed-point representation with total_bits and frac_bits.
    It assumes an accumulator with acc_bits total bits and acc_frac_bits
    fractional bits to aim at preventing overflow during intermediate calculations.

    It does not quantize the input signal.
    """

    b_arr = np.asarray(B, dtype=np.float64)
    a_arr = np.asarray(A, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    M = len(b_arr) - 1
    N = len(a_arr) - 1

    if a_arr[0] != 1:
        raise ValueError(
            "A[0] must be 1 for normalized IIR filters and also for FIR filters. Consider normalizing the filter coefficients."
        )

    y = np.zeros_like(x_arr)

    for n in range(0, len(x_arr)):
        acc = 0.0

        for i in range(M + 1):  # from B[0] to B[M]
            if n - i >= 0:
                acc += b_arr[i] * x_arr[n - i]
                _, _, acc = fixed_point_conversion(
                    acc, acc_bits, acc_frac_bits, mode="round"
                )

        for i in range(1, N + 1):  # from A[1] to A[N]
            if n - i >= 0:
                acc -= a_arr[i] * y[n - i]
                _, _, acc = fixed_point_conversion(
                    acc, acc_bits, acc_frac_bits, mode="round"
                )

        _, _, y[n] = fixed_point_conversion(acc, total_bits, frac_bits, mode="round")

    return y


def mac_int16_add(x1: np.int8, x2: np.int8, acc: np.int16) -> np.int16:
    """
    Multiply-accumulate operation for fixed-point filtering using addition.
    Expand the range of the accumulator to avoid getting wrap-around before we can saturate.
    """
    # model 8-bit multiplier producing product stored in 16-bit accumulator:
    prod = np.int16(x1) * np.int16(x2)
    # temporarilly use int32 to avoids overflow before saturation:
    new_acc_value = saturate_int16(np.int32(acc) + np.int32(prod))
    return new_acc_value


def mac_int16_sub(x1: np.int8, x2: np.int8, acc: np.int16) -> np.int16:
    """
    Multiply-accumulate operation for fixed-point filtering using subtraction.
    Expand the range of the accumulator to avoid getting wrap-around before we can saturate.
    """
    prod = np.int16(x1) * np.int16(x2)
    new_acc_value = saturate_int16(np.int32(acc) - np.int32(prod))
    return new_acc_value


def apply_fixed_point_filter_int8(
    B_i: npt.NDArray[np.int8],
    A_i: npt.NDArray[np.int8],
    x_i: npt.NDArray[np.int8],
    acc_frac_bits: int = 3,
    mode: Literal["floor", "round", "trunc"] = "round",
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    """
    This version is more specialized in comparison to the more
    general apply_fixed_point_filter() method. It assumes all inputs and coefficients are already quantized to int8 fixed-point representation,
    and it returns the output in both float and int8 formats for comparison.
    It implements a fixed-point IIR filter using Direct Form I structure with:
      - int8 for x, B, A and y
      - int16 for accumulator
    This function only supports normalized filters with Az[0] = 1, which is common for IIR filters and also applies to FIR filters.
    This models one possible fixed-point architecture: int8 inputs/states/coefs, int16 saturating accumulator, and int8 output storage.
    The internal filter state (filter memory) is also quantized to int8.
    All multiplications generate Q(2*frac_bits).
    Before storing y[n] as int8, the accumulator is shifted right by frac_bits.
    This method assumes that A_i[0] is "1" and it is not used in the calculations.
    """

    print(" B_i.dtype:", B_i.dtype, "A_i.dtype:", A_i.dtype, "x_i.dtype:", x_i.dtype)
    if B_i.dtype != np.int8:
        raise ValueError("B must be of type int8")
    if A_i.dtype != np.int8:
        raise ValueError("A must be of type int8")
    if x_i.dtype != np.int8:
        raise ValueError("x must be of type int8")

    M = len(B_i) - 1  # number of numerator coefficients
    N = len(A_i) - 1  # number of denominator coefficients

    y_i = np.zeros(len(x_i), dtype=np.int8)  # pre-allocate space for int8

    for n in range(len(x_i)):
        acc = np.int16(0)  # initialize int16 accumulator as zero

        for k in range(M + 1):  # implement numerator part of the filter
            if n - k >= 0:  # skip negative indices during transient response
                acc = mac_int16_add(B_i[k], x_i[n - k], acc)

        for k in range(1, N + 1):  # implement denominator part of the filter
            if n - k >= 0:  # skip negative indices during transient response
                # Note that for Direct Form I implementation, we have the negative sign
                # for feedback coefficients in the accumulator. If we do:
                # acc = mac_int16(-A_i[k], y_i[n - k], acc)
                # in the extreme case A_i[k] == -128, then -A_i[k] overflows in int8.
                # Because of that we will use:
                acc = mac_int16_sub(A_i[k], y_i[n - k], acc)

        # acc is in Q(2*frac_bits) but y_i must be in Q(frac_bits)
        if mode == "round":
            # trying to round to nearest to avoid having negative outputs rounded differently from positive outputs.
            if acc >= 0:
                acc_shifted = (acc + (1 << (acc_frac_bits - 1))) >> acc_frac_bits
            else:
                acc_shifted = -((-acc + (1 << (acc_frac_bits - 1))) >> acc_frac_bits)
        elif mode == "floor":
            acc_shifted = acc >> acc_frac_bits
        elif mode == "trunc":
            acc_shifted = int(acc / (1 << acc_frac_bits))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        y_i[n] = saturate_int8(acc_shifted)  # save output as int8

    y = int8_fixed_to_float(y_i, acc_frac_bits)  # convert back to float

    return y_i, y


def main_comparison_float_vs_fixed_point() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Generate plots and outputs to illustrate fixed-point roundoff effects.
    """
    np.random.seed(0)

    N = 300
    x = 10 * np.cos(np.pi * 0.2 * np.arange(N))
    # x = 32 * np.random.randn(N)

    M = 4
    butter_out = butter(M, 0.3)
    if not isinstance(butter_out, tuple) or len(butter_out) != 2:
        raise RuntimeError("scipy.signal.butter did not return (b, a)")
    b_tmp, a_tmp = butter_out
    B = np.asarray(b_tmp, dtype=np.float64)
    A = np.asarray(a_tmp, dtype=np.float64)

    # 8-bit Q4.3 variables and coefficients
    total_bits = 8
    frac_bits = 3

    # 16-bit accumulator with 6 fractional bits
    acc_bits = 16
    acc_frac_bits = 2 * frac_bits

    xq, xi = quantize_array(x, total_bits, frac_bits, mode="round")
    Bq, Bi = quantize_array(B, total_bits, frac_bits, mode="round")
    Aq, Ai = quantize_array(A, total_bits, frac_bits, mode="round")

    if np.sum(np.abs(Bq)) == 0 or np.sum(np.abs(Aq)) == 0:
        raise RuntimeError("Quantized filter has B(z) or A(z) only with zero values.")

    w, H, Hq = compare_frequency_responses(B, A, Bq, Aq)

    yq_my = apply_fixed_point_filter(
        Bq,
        Aq,
        xq,
        total_bits=total_bits,
        frac_bits=frac_bits,
        acc_bits=acc_bits,
        acc_frac_bits=acc_frac_bits,
        mode="round",
    )

    y_my = apply_fixed_point_filter(
        B,
        A,
        x,
        total_bits=64,
        frac_bits=40,
        acc_bits=64,
        acc_frac_bits=40,
        mode="round",
    )

    y_scipy = np.asarray(lfilter(B, A, x), dtype=np.float64)

    Bq_int8 = np.asarray(Bq * (2 ** frac_bits), dtype=np.int8)
    Aq_int8 = np.asarray(Aq * (2 ** frac_bits), dtype=np.int8)
    xq_int8 = np.asarray(xq * (2 ** frac_bits), dtype=np.int8)
    y_int8, _ = apply_fixed_point_filter_int8(
        Bq_int8, Aq_int8, xq_int8, acc_frac_bits=frac_bits, mode="round"
    )

    n = np.arange(N)

    plt.figure()
    plt.plot(n, yq_my, linewidth=3, label="Quantized my_filter")
    plt.plot(n, y_my, "o-", label="Float my_filter")
    plt.plot(n, y_scipy, "x-", label="scipy.signal.lfilter")
    plt.plot(n, y_int8, "s-", label="Int8 my_filter")
    plt.xlabel("n")
    plt.ylabel("Filter output y[n]")
    plt.legend()
    plt.grid(True)
    plt.show()

    return yq_my, y_my, y_scipy, Bq, Aq, xq


def fixed_point_conversion(
    x: npt.ArrayLike | float | int | np.integer | np.floating,
    b: int,
    b_f: int,
    mode: Literal["floor", "round", "trunc"] = "round",
) -> tuple[str, int, float]:
    """
    Fixed-point conversion for signed numbers with b bits:
    one bit for the sign and b_f fractional bits.
    The representable range is [-2^(b-1), 2^(b-1)-1].
    Returns: x_b, x_i, x_q
            x_b: binary representation with b bits
            x_i: x_q represented as an integer
            x_q: decoded quantized value
    alternatives for "mode":
        "floor" -> floor(-3.8) = -4 and floor(3.8) = 3
        "round" -> round to nearest integer, e.g., round(-3.8) = -4, round(3.8) = 4, round(-3.5) = -4, round(3.5) = 4
        "trunc" -> truncate toward zero, e.g., trunc(-3.8) = -3, trunc(3.8) = 3
    """
    # make sure x is a scalar and float
    x_array = np.asarray(x, dtype=np.float64)
    if x_array.size != 1:
        raise ValueError(
            "fixed_point_conversion expects a scalar. Pass one coefficient at a time."
        )
    x = float(x_array.reshape(-1)[0])

    Delta = 2 ** (-b_f)  # quantization step size

    number_of_deltas = x / Delta

    if mode == "floor":
        x_i = math.floor(number_of_deltas)
    elif mode == "round":
        x_i = round(number_of_deltas)
    elif mode == "trunc":
        x_i = int(number_of_deltas)
    else:
        raise ValueError("mode must be 'floor', 'round', or 'trunc'")

    # Scaled-integer range for signed fixed-point with 1 sign bit
    min_int = -(2 ** (b - 1))
    max_int = (2 ** (b - 1)) - 1

    if x_i < min_int:  # check minimum representable value
        x_i = min_int
        print(
            "Warning! Consider increasing number of bits to represent the integer part (b-1-b_f)"
        )

    if x_i > max_int:  # check maximum representable value
        x_i = max_int
        print(
            "Warning! Consider increasing number of bits to represent the integer part (b-1-b_f)"
        )

    x_q = x_i * Delta  # quantized value scaled back to original range

    if x_i < 0:
        # complement 2's representation for negative numbers
        # Two's complement of -N is: 2^b - N
        # Since (1 << b) = 2^b, the expression becomes: 2^b + (-N) = 2^b - N
        x_b = format((1 << b) + x_i, f"0{b}b")
    else:
        x_b = format(x_i, f"0{b}b")

    return x_b, x_i, x_q


def quantize_array(
    x: npt.ArrayLike | float | int | np.integer | np.floating,
    b: int,
    b_f: int,
    mode: Literal["floor", "round", "trunc"] = "round",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Quantize all elements of an array x to fixed-point representation.

    Parameters
    ----------
    x : array elements to quantize
    b : int
        Total number of bits for fixed-point representation (including sign bit).
    b_f : int
        Number of bits for the fractional part.
    mode : str, optional
        Rounding mode: "floor", "round", or "trunc". Default is "round".
    Returns
    -------
    coefficients_quantized : ndarray
        Quantized coefficients.
     Note: For IIR filters, A[0] is typically 1 for normalized filters, so it may not need quantization.
    """
    x_array = np.atleast_1d(np.asarray(x, dtype=np.float64))
    x_quantized: list[float] = []
    x_as_integers: list[int] = []
    for value in np.asarray(x_array, dtype=np.float64).reshape(-1).tolist():
        _, x_i, x_q = fixed_point_conversion(value, b, b_f, mode=mode)
        x_quantized.append(x_q)
        x_as_integers.append(x_i)

    return np.array(x_quantized, dtype=np.float64), np.array(
        x_as_integers, dtype=np.int64
    )


def main_quantization_examples():
    # Example
    input_values = [5.0625, 0.6328125, -7.45, 2804.6542]
    b_f_values = [4, 7, 4, 3]
    b_values = [8, 8, 8, 16]
    for x, b, b_f in zip(input_values, b_values, b_f_values):

        print(
            "\nInput: x =",
            x,
            ", b =",
            b,
            ", b_f =",
            b_f,
            ", # bits for integer part m =",
            b - 1 - b_f,
        )
        x_b, x_i, x_q = fixed_point_conversion(x, b, b_f)

        print("x_b =", x_b)
        print("x_i =", x_i)
        print("x_q =", x_q)
        print("x =", x)
        print("error=x-x_q =", x - x_q)


if __name__ == "__main__":
    main_quantization_examples()

    yq_my, y_my, y_scipy, Bq, Aq, xq = main_comparison_float_vs_fixed_point()
    print("Quantized B =", Bq)
    print("Quantized A =", Aq)
