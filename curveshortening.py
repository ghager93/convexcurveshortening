import numpy as np

from scipy import ndimage
from scipy import interpolate
from typing import List

import _scaling_functions
import _metrics
import _vector_maths


def convex_curve_shortening_flow(curve: np.ndarray,
                                 precision: int = 100,
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
    edge_length_init = _vector_maths.edge_length(curve)
    resampling_factor = n_vertices_init / edge_length_init.sum()

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

    concave_curves = _reduce_concave_iterations_to_precision(curves, precision)

    convex_curves = curve_shortening_flow(concave_curves[-1], precision - len(concave_curves))

    return concave_curves + convex_curves


def curve_shortening_flow(curve: np.ndarray, n_curves: int, return_initial_curve=False):
    '''
    Curve shortening flow.

    :param curve: Nx2 Numpy array, where N is the number of vertices in the curve.
    :param n_curves: Number of curve iterations between the original and singularity.
    :return:
    '''

    if type(curve) is not np.ndarray or curve.ndim != 2:
        raise ValueError('Curve must be 2-D Numpy array.')

    if curve.shape[0] < 3:
        raise ValueError('Curve must have at least 3 vertices.')

    if curve.shape[1] != 2:
        raise ValueError('Curve must have the shape Nx2, i.e. N rows of 2D coordinates.')

    n_vertices = curve.shape[0]
    linear_stds = _linear_step_stds(curve, n_curves, startpoint=return_initial_curve)

    curves = [_mokhtarian_mackworth92(curve, sigma) for sigma in linear_stds]

    return curves


def _linear_step_stds(curve: np.ndarray, n_curves: int, startpoint=True):
    # Array of n_curve stds that will create linearly spaced curves.
    # Curve shortening flow algorithm found to closely match scaled normal distribution;
    # N(x; \sigma) = n_vertices \exp(-x^2 / 2\sigma^2),    \sigma = pi/20, x > 0

    sigma2 = (np.pi/20)**2
    average_radius = _metrics.average_radius(curve)

    if startpoint:
        linear_steps = np.linspace(average_radius, 0, n_curves, endpoint=False)
    else:
        linear_steps = np.linspace(average_radius, 0, n_curves + 1, endpoint=False)[1:]
    return curve.shape[0] * np.sqrt(2 * sigma2 * np.log(average_radius / linear_steps))


def _mokhtarian_mackworth92(curve, sigma):
    # Mokhtarian, Farzin & Mackworth, Alan. (1992).
    # A Theory of Multiscale, Curvature-Based Shape Representation for Planar Curves.
    # Pattern Analysis and Machine Intelligence, IEEE Transactions on. 14. 789-805. 10.1109/34.149591.

    # Apply Gaussian filter, followed by resampling.

    # return _resample(_gaussian_filter(curve, sigma), 1 / _edge_length(curve).mean())
    return _gaussian_filter(curve, sigma)


def _magnitude_array(curve: np.ndarray):
    # Magnitudes of iteration for each vertex.
    # Calculated as a scaled version of the normalised curvature.

    return _scale_curvature(_metrics.normalised_curvature(curve))


def _vector_array(curve: np.ndarray):
    # Vectors of iteration for each vertex.
    # Calculated as parallel to the normal and facing inward.

    return _vector_maths.inward_normal(curve)


def _scale_curvature(curvature: np.ndarray):
    return _scaling_functions.f_sigmoid(10, 0.1)(curvature)


def _break_condition(curve: np.ndarray):
    return _metrics.concavity(curve) < 0.1


def _break_condition_convex(curve):
    return curve.shape[0] < 4


def _gaussian_filter(curve: np.ndarray, sigma: float):
    return ndimage.gaussian_filter1d(curve, sigma, axis=0, mode='wrap')


def _resample_condition(curve: np.ndarray):
    return True


def _resample(curve: np.ndarray, factor: float):
    # Resample the vertices along the curve.
    # Return the same curve but with n=int(factor * curve_length) equidistant vertices.

    current_lengths = _vector_maths.edge_length(curve)
    curve_looped = np.vstack((curve, curve[0]))
    cumulative_lengths_zero_start = np.hstack((0, current_lengths.cumsum()))
    total_length = cumulative_lengths_zero_start[-1]

    interp_func = interpolate.interp1d(cumulative_lengths_zero_start, curve_looped, axis=0)
    new_lengths = np.linspace(0, total_length, int(factor * total_length), endpoint=False)

    return interp_func(new_lengths)


def _mean_distance_to_centre_of_mass(curve: np.ndarray):
    # Metric for determining number of iterations to singularity.
    # d = 1/|curve| * sum_s(||curve(s) - 1/|curve| * sum_s'(curve(s')) ||)

    return np.linalg.norm(curve-curve.mean(axis=0), axis=1).mean()


def _reduce_concave_iterations_to_precision(curves: List, precision: int):
    return _reduce_to_roughly_equal_curves(curves, precision)


def _reduce_to_roughly_equal_curves(curves: List, precision: int):
    # Reduces number of concave curve iterations in list to be proportional to desired precision.
    # Includes the first and last curves.
    # Proportion of total iterations that are concave is determined as one minus the ratio
    # of the length of the last concave curve to the length of the first.
    # Thus, the number of returned curves is floor(precision * (1 - curves[-1] / curves[0])).

    n_curves = np.floor(precision *
                        (1 - _metrics.total_edge_length(curves[-1]) / _metrics.total_edge_length(curves[0]))
                        ).astype(int)

    return [curves[i] for i in np.round(np.linspace(0, len(curves) - 1, n_curves)).astype(int)]
