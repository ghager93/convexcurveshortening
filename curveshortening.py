import numpy as np

from scipy import ndimage
from scipy import interpolate

import _scaling_functions


def convex_curve_shortening_flow(curve: np.ndarray,
                                 step_size: float = 1,
                                 step_sigma: float = 10,
                                 resample_sigma: float = 1,
                                 scaling_function_type: str = "sigmoid",
                                 scaling_function_alpha: float = None,
                                 scaling_function_a: float = None):
    '''
    Convex curve shortening.

    :param scaling_function: Function type used for scaling the curvature magnitude vector.
    :param resample_sigma: Standard deviation for the Gaussian filter used during resampling.
    :param step_sigma: Standard deviation for the Gaussian filter used on the step vector.
    :param curve: Nx2 Numpy array, where N is the number of vertices in the curve.
    :param step_size: Scales magnitude of each iteration.
    :return:
    '''

    if type(curve) is not np.ndarray or curve.ndim != 2:
        raise ValueError('Curve must be 2-D Numpy array.')

    if curve.shape[0] < 3:
        raise ValueError('Curve must have at least 3 vertices.')

    if curve.shape[1] != 2:
        raise ValueError('Curve must have the shape Nx2, i.e. N rows of 2D coordinates.')

    if scaling_function_type == "sigmoid":
        scaling_func = _scaling_functions.f_sigmoid(10, 0.1)
    elif scaling_function_type == "elu":
        scaling_func = _scaling_functions.f_elu(1, -0.1)
    elif scaling_function_type == "softplus":
        scaling_func = _scaling_functions.f_softplus(1, 0.1)
    else:
        scaling_func = _scaling_functions.f_sigmoid(10, 0.1)

    max_iterations = 10000

    n_vertices_init = curve.shape[0]
    edge_length_init = _edge_length(curve)
    resampling_factor = n_vertices_init / edge_length_init

    curves = [curve]

    for i in range(max_iterations):

        if _break_condition(curve):
            break

        if _resample_condition(curve):
            curve = _gaussian_filter(_resample(curve, resampling_factor), resample_sigma)

        curve = curve.astype(float)

        step_vectors = step_size * _magnitude_array(curve)[:, None] * _vector_array(curve)

        curve_new = curve + _gaussian_filter(step_vectors, step_sigma)

        curve = curve_new

        curves.append(curve)

    return curves


def _magnitude_array(curve: np.ndarray):
    # Magnitudes of iteration for each vertex.
    # Calculated as a scaled version of the normalised curvature.

    return _scale_curvature(_normalise_curvature(_curvature(curve)))


def _vector_array(curve: np.ndarray):
    # Vectors of iteration for each vertex.
    # Calculated as parallel to the normal and facing inward.

    return _inward_normal(curve)


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


def _edge_length(curve: np.ndarray):
    # Euclidean distance between each neighbouring vertex.

    return np.linalg.norm(curve - np.roll(curve, 1, axis=0), axis=1)


def _tangent(curve: np.ndarray):
    return (curve - np.roll(curve, 1, axis=0)) / _edge_length(curve)[:, None]


def _normal(curve: np.ndarray):
    tangent = _tangent(curve)
    edge_length = _edge_length(curve)

    return 2 * ((np.roll(tangent, -1, axis=0) - tangent) /
                (np.roll(edge_length, -1, axis=0) + edge_length)[:, None])


def _inward_normal(curve: np.ndarray):
    # The inward normal is the normal pointing toward the interior of a closed curve.
    # It is the tangent vector rotated clockwise 90 degrees.

    return _tangent(curve) @ np.array([[0, -1], [1, 0]])


def _concavity(curve: np.ndarray):
    curvature = _curvature(curve)

    return -sum(curvature[curvature < 0])


def _break_condition(curve: np.ndarray):
    return _concavity(curve) < 0.1


def _gaussian_filter(curve: np.ndarray, sigma: float):
    return ndimage.gaussian_filter1d(curve, sigma, axis=0, mode='wrap')


def _resample_condition(curve: np.ndarray):
    return True


def _resample(curve: np.ndarray, factor: float):
    # Resample the vertices along the curve.
    # Return the same curve but with n=int(factor * curve_length) equidistant vertices.

    current_lengths = _edge_length(curve)
    interp_func = interpolate.interp1d(current_lengths.cumsum()-current_lengths[0], curve, axis=0)
    new_lengths = np.linspace(0, current_lengths.sum())
    return interp_func(np.linspace(current_lengths[0], current_lengths.sum(),
                                   int(factor * current_lengths.sum())))
