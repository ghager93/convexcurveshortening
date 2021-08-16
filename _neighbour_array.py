import numpy as np
from functools import wraps

NEIGHBOUR_REF = np.array([[7, 0, 1],
                          [6, 8, 2],
                          [5, 4, 3]])


def binary_to_array(b: int):
    arr = np.zeros((3, 3), dtype='bool')
    arr[0, 0] = b & 0b10000000
    arr[0, 1] = b & 0b00000001
    arr[0, 2] = b & 0b00000010
    arr[1, 0] = b & 0b01000000
    arr[1, 1] = 0
    arr[1, 2] = b & 0b00000100
    arr[2, 0] = b & 0b00100000
    arr[2, 1] = b & 0b00010000
    arr[2, 2] = b & 0b00001000

    return arr.astype(int)


def array_to_binary(array: np.ndarray):
    assert array.shape == (3, 3)

    return array[0, 1] + (array[0, 2] << 1) + (array[1, 2] << 2) + (array[2, 2] << 3) \
           + (array[2, 1] << 4) + (array[2, 0] << 5) + (array[1, 0] << 6) + (array[0, 0] << 7)


def second_neighbours_array_to_binary(arr: np.ndarray):
    assert arr.shape == (5, 5)

    mask = np.array([[2 ** 15, 1, 2, 4, 8],
                     [2 ** 14, 0, 0, 0, 16],
                     [2 ** 13, 0, 0, 0, 32],
                     [2 ** 12, 0, 0, 0, 64],
                     [2 ** 11, 1024, 512, 256, 128]])

    return np.sum(mask * arr)


def get_neighbour_array(array: np.ndarray):
    shifted_neighbour_arrays = get_shifted_neighbour_arrays(array)
    neighbour_array = np.zeros(array.shape, dtype='int')
    neighbour_bit_shifts = [7, 0, 1, 6, 2, 5, 4, 3]
    for i, arr in enumerate(shifted_neighbour_arrays):
        neighbour_array |= arr << neighbour_bit_shifts[i]

    return neighbour_array


def get_neighbour_array_no_hang(array: np.ndarray):
    out = get_neighbour_array(array)
    out[array == 0] = 0
    return out


def get_shifted_neighbour_arrays(array: np.ndarray):
    padded_array = np.pad(array, 1)
    return [padded_array[x:padded_array.shape[0] + x - 2, y:padded_array.shape[1] + y - 2]
            for x in range(3) for y in range(3)
            if x != 1 or y != 1]


def _array_to_binary_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [array_to_binary(a) if type(a) is np.ndarray else a for a in args]
        kwargs = {kw: (array_to_binary(a) if type(a) is np.ndarray else a) for (kw, a) in kwargs.items()}
        return func(*args, **kwargs)

    return wrapper


@_array_to_binary_wrapper
def number_of_neighbours(n):
    return _number_of_bits_high(n)


def number_of_second_neighbours(arr: np.ndarray):
    assert arr.shape == (5, 5)

    mask = np.array([[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]])

    return np.count_nonzero(mask & arr)


def _number_of_bits_high(b: int):
    return bin(b).count('1')


@_array_to_binary_wrapper
def number_of_connected_neighbours(b):
    return _number_of_01_patterns_in_ordered_neighbours_set(b)


@_array_to_binary_wrapper
def number_of_connected_second_neighbours(b):
    return _number_of_01_patterns_in_ordered_second_neighbours_set(b)


def _number_of_01_patterns_in_ordered_neighbours_set(b: int):
    mask = 0b11000000
    pattern = 0b01000000
    cnt = 0
    for _ in range(7):
        if mask & b == pattern:
            cnt += 1
        mask >>= 1
        pattern >>= 1

    if 0b10000001 & b == 0b10000000:
        cnt += 1

    return cnt


def _number_of_01_patterns_in_ordered_second_neighbours_set(b):
    mask = 0b1100000000000000
    pattern = 0b0100000000000000
    cnt = 0
    for _ in range(15):
        if mask & b == pattern:
            cnt += 1
        mask >>= 1
        pattern >>= 1

    if 0b1000000000000001 & b == 0b1000000000000000:
        cnt += 1

    return cnt


@_array_to_binary_wrapper
def is_top_left_corner_of_square(b):
    mask = 0b00011100
    return mask & b == mask


@_array_to_binary_wrapper
def number_of_side_neighbours(b):
    mask = 0b01010101
    return bin(mask & b).count('1')


@_array_to_binary_wrapper
def number_of_diagonal_neighbours(b):
    mask = 0b10101010
    return bin(mask & b).count('1')


@_array_to_binary_wrapper
def neighbour_coordinates(b):
    return side_neighbour_coordinates(b) + diagonal_neighbour_coordinates(b)


@_array_to_binary_wrapper
def side_neighbour_coordinates(b):
    coords = set()

    if b & 0b00000001:
        coords.add((-1, 0))
    if b & 0b00000100:
        coords.add((0, 1))
    if b & 0b00010000:
        coords.add((1, 0))
    if b & 0b01000000:
        coords.add((0, -1))

    return coords


@_array_to_binary_wrapper
def diagonal_neighbour_coordinates(b):
    coords = set()

    if b & 0b00000010:
        coords.add((-1, 1))
    if b & 0b00001000:
        coords.add((1, 1))
    if b & 0b00100000:
        coords.add((1, -1))
    if b & 0b10000000:
        coords.add((-1, -1))

    return coords


def relative_neighbour_binary_dep(point: np.ndarray, neighbour: np.ndarray):
    return array_point_to_binary(neighbour - point)


def array_point_to_binary_dep(point: np.ndarray):
    return NEIGHBOUR_REF[point + (1, 1)]


def relative_neighbour_binary(points: np.ndarray, neighbours: np.ndarray):
    return array_point_to_binary(neighbours - points)


def array_point_to_binary(points: np.ndarray):
    return NEIGHBOUR_REF[tuple((points + (1, 1)).transpose())]
