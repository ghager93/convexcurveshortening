import numpy as np

import _neighbour_array


class ImageCurve():
    def __init__(self, im: np.ndarray):
        self._im = np.pad(im, 1)
        self._im_curve = _edge_detect(self._im)
        self._visited = set()
        self._start = self._starting_point()

    def _starting_point(self):
        # List of vertices starts at the vertex with minimum index (most top-left corner vertex).

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
        if starting_neighbourhood[2, 2]:
            n = 1, 1
        if starting_neighbourhood[2, 1]:
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
        [neighbours.append(point + n) for n in _neighbour_array.side_neighbour_coordinates(neighbourhood) +
         _neighbour_array.diagonal_neighbour_coordinates(neighbourhood)]

        neighbours = [n for n in neighbours if n not in self._visited]

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

        curve = [self._start]
        self._visited = set()
        self._visited.add(self._start)
        self._visited.add(self._last_point())

        curr = self._second_point()
        while curr:
            self._visited.add(curr)
            curve.append(curr)
            curr = self._next_neighbour(curr)
        curve.append(self._last_point())

        return self._unpad(np.array(curve))


def _edge_detect(array: np.ndarray):
    # Binary edge detection.  Matrix is padded and XOR'd with the intersection of shifted versions of itself.
    # Shifts are north, east, south and west.

    array = np.pad(array, 1)
    return array[1:-1, 1:-1] & np.invert(array[1:-1, :-2] & array[1:-1, 2:] &
                                         array[:-2, 1:-1] & array[2:, 1:-1])
