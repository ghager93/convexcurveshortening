import numpy as np

from typing import Collection

KERNEL_FILL_VALUE = 1
OUTPUT_FILL_VALUE = 1
KERNEL_ARRAY_DTYPE = 'int32'
OUTPUT_ARRAY_DTYPE = 'int32'


class StructuringElement:
    def __init__(self, kernel: np.ndarray, centre: Collection):
        self.kernel = kernel
        self.centre = centre

    def width(self):
        return self.kernel.shape[0]

    def height(self):
        return self.kernel.shape[1]


def circle(radius: int = 1):
    return StructuringElement(_make_circular_kernel(radius), (radius, radius))


def _make_circular_kernel(radius: int):
    filter_shape = _get_kernel_shape(radius)
    circular_kernel = np.zeros(filter_shape, dtype=KERNEL_ARRAY_DTYPE)
    xx, yy = np.mgrid[:filter_shape[0], :filter_shape[1]]
    circular_kernel[(xx - radius) ** 2 + (yy - radius) ** 2 <= radius ** 2] = KERNEL_FILL_VALUE
    return circular_kernel


def _get_kernel_shape(radius: int):
    return 2 * radius + 1, 2 * radius + 1


def _assert_structuring_element_smaller_than_or_equal_to_array(array: np.ndarray,
                                                               structuring_element: StructuringElement):
    assert array.shape[0] >= structuring_element.kernel.shape[0] \
           and array.shape[1] >= structuring_element.kernel.shape[1]
