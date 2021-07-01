import numpy as np

import _scaling_functions

def convex_curve_shortening_flow(curve: np.ndarray):
    '''
    Convex curve shortening.

    :param curve: Nx2 Numpy array, where N is the number of vertices in the curve.
    :return:
    '''

    if type(curve) is not np.ndarray or curve.ndim != 2:
        raise ValueError('Curve must be 2-D Numpy array.')

    if curve.shape[0] < 3:
        raise ValueError('Curve must have at least 3 vertices.')

    if curve.shape[1] != 2:
        raise ValueError('Curve must have the shape Nx2, i.e. N rows of 2D coordinates.')

    curve = curve.astype(float)


def _curvature_magnitude_array(curve: np.ndarray):
    pass


def _curvature(curve: np.ndarray):
    # Curvature is the cross-product of the normal and the tangent.
    # k = t x n
    #   = x'y" - x"y'

    tangent = _tangent(curve)
    normal = _normal(curve)

    return ((tangent[:, 1] * normal[:, 0] - normal[:, 1] * tangent[:, 0])
            / ((tangent[:, 1] ** 2 + tangent[:, 0] ** 2) ** (3 / 2)))


def _normalise_curvature(curvature: np.ndarray):
    return curvature / np.max(abs(curvature))


def _scale_curvature(curvature: np.ndarray):
    return _scaling_functions.f_sigmoid(10, 0.1)(curvature)


def _edge_length_euc(curve):
    # Euclidean distance between each neighbouring vertex.

    return np.linalg.norm(curve - np.roll(curve, 1, axis=0), axis=1)


def _tangent(curve):
    return (curve - np.roll(curve, 1, axis=0)) / _edge_length_euc(curve)[:, None]


def _normal(curve):
    tangent = _tangent(curve)
    edge_length = _edge_length_euc(curve)

    return 2 * ((np.roll(tangent, -1, axis=0) - tangent) /
                (np.roll(edge_length, -1, axis=0) + edge_length)[:, None])


def inward_normal(curve):
    # The inward normal is the normal pointing toward the interior of a closed curve.
    # It is the tangent vector rotated clockwise 90 degrees.

    return _tangent(curve) @ np.array([[0, -1], [1, 0]])


def _concavity(curve):
    curvature = _curvature(curve)

    return -sum(curvature[curvature < 0])


def _break_condition(curve: np.ndarray):
    return _concavity(curve) < 0.1

