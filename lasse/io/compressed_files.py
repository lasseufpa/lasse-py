"""
The methods allow to write binary files using less than 8 bits per sample.
The motivation is that programming languages support reading/writing bytes (8 bits)
and its multiples, while here we support b < 8 bits per sample.
Details:
We pack N samples of b bits/sample into bytes in a bit stream such that the
output file has a minimum size and a 1-byte header. In other words, the output
file uses a single byte as "header". The header indicates the number H of b-bits
samples that will be used to complete the trailing byte in case N * b is not a
multiple of 8 bits. These trailing bits are set as zeros. The endianess is "little".
The method compact_bytes "squishes" the data together ignoring the zeros on the most
significant bits, while decompact_bytes undoes this process, introducing bits "zeros"
to use b bits per sample.

For instance, an input array with N = 8 samples:
    x = np.array([3, 2, 1, 0, 3, 2, 1, 3], dtype=np.int64)
and num_bits=2 bits per sample, is written as a file with (N * b + 8) = (8 * 2 + 8) =
24 bits = 3 bytes. These 3 bytes are:
00 1b db in hexa, which corresponds, in bits, to 0000 0000  0001 1011  1101 1011
where the first byte is H=00 hexa because the 8 samples of 2 bits each exactly fits
2 bytes, so H=0 samples of b bits each are needed to complete an integer number of bytes.
Due to the assumed little endianness the mapping of integer values and bits are:
0000 0000 | 00 01 10 11 | 11 01 10 11
header=0  |  0  1  2  3 |  3  1  2  3
can be represented with 2 bits per sample (that support the range [0, 3])

As another example, consider the input with N=4 samples
    x = np.array([7, 1, 2, 3], dtype=np.int64)
    num_bits = 3
is also written as a file with the 3 bytes (4 * 3 + 8 = 20 bits + 4 trailing bits to
complete a byte). These 3 bytes are:
01 8f 06 in hexa, which corresponds, in bits, to 0000 0001  1000 1111  0000 0110
where the first byte is H=01 hexa, which corresponds to H=1 samples of 3-bits.
Note that we need 4 trailing bits in this case, but H=1 sample. We would have H=2 only
when using 6 or 7 trailing bits given that H is a floor value for the number of b-bits
sampled. In this example, due to the assumed little endianness the mapping of
integer values and bits are:
0000 0001 |  10                001 111 | 0 000 011  0
header=1  |  suffix of 010=2     1   7 |         3  prefix of 010=2

@authors Eduardo Filho, 2023
Reviewed by Aldebaro, 2025
"""

import numpy as np

'''
Encode input_array x into a bit array using num_bits per sample, where
num_bits should be an integer smaller than 8.
'''
def compact_bytes(input_array, num_bits) -> np.ndarray:
    if num_bits >= 8:
        raise ValueError("This function is meant to work with less than 8 bits!")

    min = np.min(input_array)
    max = np.max(input_array)
    if min < 0 or max > 2**num_bits:
        max_allowed = 2**num_bits-1
        raise ValueError("The input must be in the range [0, " +
                         str(max_allowed) + "] but I found min, max =", min, max)

    if not np.issubdtype(input_array.dtype, np.integer):
        print("Found values: ", input_array)
        raise ValueError("Input must be integer!")

    input_arr_len = len(input_array)
    bit_array = np.array([], dtype=np.uint8)

    # For every value in the array, open it into a bit_array
    # remove the unnecessary space then save the bits into a new array
    for i in range(input_arr_len):
        tmp = np.uint8(input_array[i])
        tmp_array = np.unpackbits(tmp, bitorder="little")[0:num_bits]
        bit_array = np.concatenate((bit_array, tmp_array), dtype=np.uint8)

    # Put the bits into a new array, np.packbits does the "heavylifting"
    output_array = np.packbits(bit_array, bitorder="little")

    # Calculate the number of zeros introduced when np.packbits leaves trailing zeros
    # in a byte and save it in the first index of the array
    estimated_decompressed_array_len = (len(output_array) * 8) // num_bits
    num_discrepant_zeros = estimated_decompressed_array_len - input_arr_len

    # For debugging
    print("num_discrepant_zeros =", num_discrepant_zeros)

    return_array = np.insert(output_array, obj=0, values=num_discrepant_zeros)

    return np.array(return_array)  # Needed to cast to np.array to pass pyright

'''
Decode input_array into a uint8 array using num_bits per sample, where
num_bits should be an integer smaller than 8.
'''
def decompact_bytes(input_array, num_bits) -> np.ndarray:
    if num_bits >= 8:
        raise ValueError("This function is meant to work with less than 8 bits!")
    
    # Separate the data from the "header" containing the number of discrepant zeros
    num_discrepant_zeros = input_array[0]
    data_array = input_array[1:]

    # Calculate the number of elements present in the original array, which
    # must be the same in the decompressed one
    output_arr_len = (len(data_array) * 8) // num_bits - num_discrepant_zeros

    # Open the array of data into an array of bits
    bit_array = np.unpackbits(data_array, bitorder="little")
    output_array = np.array([], dtype=np.uint8)

    # For every index which should be on the decompressed array
    # separate the bits which represent that number, pack them back
    # into a byte ( uint8 ) and then put the number into a new array
    for i in range(output_arr_len):
        tmp_array = bit_array[num_bits * i : num_bits * (i + 1)]
        value = np.packbits(tmp_array, bitorder="little")

        output_array = np.concatenate((output_array, value))

    return output_array


'''
Encode the signal x into a file called filename using num_bits per sample.
x must be integer-valued and num_bits should be an integer smaller than 8.
'''
def write_encoded_file(x, num_bits, filename):
    compressed = compact_bytes(x, num_bits)
    compressed.tofile(filename)
    return compressed

'''
Read a signal x from a file called filename using num_bits per sample.
The returned numpy array x has dtype=np.uint8 and num_bits should be an integer smaller than 8.
'''
def read_encoded_file(filename, num_bits):
    compressed = np.fromfile(filename, dtype=np.uint8, count=-1)
    uncompressed = decompact_bytes(compressed, num_bits)
    return uncompressed


'''
Test methods.
'''
def main():
    filename = 'test.bin'
    #define 3 tests below
    #x = np.array([0, 1, 2, 3, 4, 5, 6, 10, 11, 15, 15, 15], dtype=np.int64)
    #num_bits = 8
    #x = np.array([3, 2, 1, 0, 3, 2, 1, 3], dtype=np.int64)
    #num_bits = 2
    x = np.array([1, 1, 1], dtype=np.int64)
    num_bits = 3
    write_encoded_file(x, num_bits, filename)

    x_recovered = read_encoded_file(filename, num_bits)
    print("Original x", x)
    print("Reconstructed x", x_recovered)
    print("Maximum error =", np.max(x-x_recovered))

if __name__ == "__main__":
    # use it for testing
    main()