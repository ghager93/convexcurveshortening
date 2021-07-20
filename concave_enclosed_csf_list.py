import numpy as np

from typing import Callable

import _scaling_functions
import _metrics
import _vector_maths
import _utils
import _refresher_classes
import _saver_classes
import _terminator_classes


class ConcaveEnclosedCSFList:
    def __init__(self, curve: np.ndarray,
                 precision: int = 100,
                 step_size: float = 1,
                 step_sigma: float = 10,
                 resample_sigma: float = 1,
                 scaling_function: Callable = None,
                 max_iterations: int = 10000):

        self.curves = list(curve)
        self.precision = precision
        self.step_size = step_size
        self.step_sigma = step_sigma
        self.resample_sigma = resample_sigma
        self.max_iterations = max_iterations

        if scaling_function is None:
            self.scaling_function = _scaling_functions.f_sigmoid(10, 0.1)

    def _set_refresher(self):
        return _refresher_classes.IterativeECSFRefresher(int(self.max_iterations / 10))

    def _set_saver(self):
        return _saver_classes.IterativeECSFRefresher(int(self.max_iterations / 10))

    def _set_iterative_terminator(self):
        pass

    def _set_time_terminator(self):
        pass

    def _set_conditional_terminator(self):
        pass

    def run():
        '''
        Enclosed curve shortening flow.

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

        if scaling_function is None:
            scaling_func = _scaling_functions.f_sigmoid(10, 0.1)

        max_iterations = 10000

        n_vertices_init = curve.shape[0]
        edge_length_init = _vector_maths.edge_length(curve)
        resampling_factor = n_vertices_init / edge_length_init.sum()

        curves = [curve]

        for i in range(max_iterations):



    def _step(self, curve):
        if _break_condition_ecsf(curve):
            break

        if _resample_condition(curve):
            curve = _utils.gaussian_filter(_utils.resample(curve, resampling_factor), resample_sigma)

        curve = curve.astype(float)

        step_vectors = step_size * _magnitude_array(curve)[:, None] * _vector_array(curve)

        curve_new = curve + _utils.gaussian_filter(step_vectors, step_sigma)

        curve = curve_new
        curves.append(curve)

    def _break_condition(self):
        return _metrics.concavity(curve) < 0.1

    def _isAlgorithmFinished(self):
        pass




