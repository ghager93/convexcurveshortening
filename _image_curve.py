import numpy as np

import skgeom

from skimage import morphology
from typing import Tuple

import _neighbour_array
import _image_processing

class ImageCurve:
    def __init__(self, im: np.ndarray):
        self._prepare_im(im)
        self._im_curve = _edge_detect(self._im)
        self._visited = set()
        self._start = self._starting_point()

    def _prepare_im(self, im):
        # Flatten values such that image is binary.
        self._im = np.where(im, 1, 0)

        # If background (indicated by top-left pixel) is 1, invert image.
        self._im = self._im ^ self._im[0, 0]

        # Added single pad layer to prevent edge cases in search.
        self._im = np.pad(self._im, 1)


    def _starting_point(self):
        # List of vertices starts at the vertex with minimum index (most top-left corner vertex).

        if not self._im.any():
            return np.array([])

        return np.argwhere(self._im)[0]

    def _second_point(self):
        # List moves clockwise with next vertex preference in order: East, South-East, South.
        # i.e.
        # [ ][ ][ ]
        # [ ][x][1]
        # [ ][3][2]

        starting_neighbourhood = self._neighbourhood(self._start)
        n = 0, 0

        if starting_neighbourhood[1, 2]:
            n = 0, 1
        elif starting_neighbourhood[2, 2]:
            n = 1, 1
        elif starting_neighbourhood[2, 1]:
            n = 1, 0

        return self._start + n

    def _last_point(self):
        # Last point preference order: South-West, South, South-East
        # [ ][ ][ ]
        # [ ][x][ ]
        # [1][2][3]

        starting_neighbourhood = self._neighbourhood(self._start)
        n = 0, 0
        if starting_neighbourhood[2, 0]:
            n =  1, -1
        if starting_neighbourhood[2, 1]:
            n = 1, 0
        if starting_neighbourhood[2, 2]:
            n = 1, 1

        return self._start + n

    def _neighbourhood(self, point: np.ndarray):
        # Returns 3x3 matrix of points in image around given point.

        return self._im_curve[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2]

    def _next_neighbour(self, point: np.ndarray):
        neighbours = list()
        neighbourhood = self._neighbourhood(point)
        # [neighbours.append(point + n) for n in neighbour_array.side_neighbour_coordinates(neighbourhood)]
        # [neighbours.append(point + n) for n in neighbour_array.diagonal_neighbour_coordinates(neighbourhood)]
        [neighbours.append(point + n) for n in _neighbour_array.side_neighbour_coordinates(neighbourhood) |
         _neighbour_array.diagonal_neighbour_coordinates(neighbourhood)]

        neighbours = [n for n in neighbours if tuple(n) not in self._visited]

        return neighbours[0] if neighbours else None

    def start(self):
        # Returns the starting index of the vertex list.

        return self._unpad(self._start)

    def _unpad(self, point: np.ndarray):
        return point - (1, 1)

    # def _unpad_list(self, list):
    #     return [p - (1, 1) for p in list]

    def curve(self):
        #  Create list of vertices, starting at the top-leftmost vertex and traversing clockwise using DFS.

        if not self._im.any():
            return np.array([])

        curve = [self._start]
        self._visited = set()
        self._visited.add(tuple(self._start))
        self._visited.add(tuple(self._last_point()))

        curr = self._second_point()
        while curr is not None:
            self._visited.add(tuple(curr))
            curve.append(curr)
            curr = self._next_neighbour(curr)
        curve.append(self._last_point())

        return self._unpad(np.array(curve))


def _edge_detect(array: np.ndarray) -> np.ndarray:
    # Binary edge detection.  Matrix is padded and XOR'd with the intersection of shifted versions of itself.
    # Shifts are north, east, south and west.

    array = np.pad(array, 1)
    return array[1:-1, 1:-1] & np.invert(array[1:-1, :-2] & array[1:-1, 2:] &
                                         array[:-2, 1:-1] & array[2:, 1:-1])


def curve_to_image_matrix(curve: np.ndarray, shape: Tuple) -> np.ndarray:
    # Convert a closed curve, defined by a list of coordinates, to a matrix of pixels.
    # Matrix is of size (shape), if this is big enough to fit the curve.
    # Otherwise, a bounding box with one pixel padding is used

    polygon = skgeom.Polygon(curve)
    bbox = polygon.bbox()
    correction = 0, 0

    if (bbox.xmin() <= 0) | (bbox.xmax() >= shape[0] - 1) | (bbox.ymin() <= 0) | (bbox.ymax() >= shape[1] - 1):
        shape = int(bbox.xmax() - bbox.xmin() + 3), int(bbox.ymax() - bbox.ymin() + 3)
        correction = int(bbox.xmin()-1), int(bbox.ymin()-1)

    image_matrix = np.zeros(shape)
    image_matrix[tuple(np.floor(p).astype(int) for p in zip(*(curve - correction)))] = 1

    return image_matrix


def curve_to_image_matrix_filled(curve: np.ndarray, shape: Tuple):
    # Curve is converted to a matrix of pixels.
    # It is then dilated by a 3x3 cross kernel to close the curve for filling.
    # The matrix curve is then filled using the flood fill technique.

    image_matrix = curve_to_image_matrix(curve, shape)
    dilated_image_matrix = morphology.dilation(image_matrix)

    return _image_processing.flood_fill(dilated_image_matrix)


def crop(curve: np.ndarray, shape: Tuple):
    # Remove points of curve outside the range [0, shape[0]-1]x[0, shape[1]-1]

    return np.array([p for p in curve if (0 <= p[0] < shape[0]) and (0 <= p[1] < shape[1])])
