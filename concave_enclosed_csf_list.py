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
    """
    Enclosed curve shortening flow.

    :param scaling_function: Function type used for scaling the curvature magnitude vector.
    :param resample_sigma: Standard deviation for the Gaussian filter used during resampling.
    :param step_sigma: Standard deviation for the Gaussian filter used on the step vector.
    :param curve: Nx2 Numpy array, where N is the number of vertices in the curve.
    :param step_size: Scales magnitude of each iteration.
    :return:
    """
    def __init__(self, curve: np.ndarray,
                 return_size: int = 100,
                 step_size: float = 1,
                 step_sigma: float = 10,
                 resample_sigma: float = 1,
                 scaling_function: Callable = None,
                 max_iterations: int = 10000,
                 max_seconds: float = 100,
                 concavity_threshold: float = 0.1,
                 refresh_interval: int = 100,
                 save_interval: int = 100):

        curve = curve.astype(float)

        self.initial_curve = curve

        self.curves = [curve]
        self.return_size = return_size
        self.step_size = step_size
        self.step_sigma = step_sigma
        self.resample_sigma = resample_sigma
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.concavity_threshold = concavity_threshold
        self.refresh_interval = refresh_interval
        self.save_interval = save_interval

        if scaling_function is None:
            self.scaling_function = _scaling_functions.f_sigmoid(10, 0.1)
        else:
            self.scaling_function = scaling_function

        self.resampling_factor = curve.shape[0] / _metrics.total_edge_length(curve)
        self.initial_area = _metrics.enclosed_area(curve)

        self.areas = [_metrics.enclosed_area(curve)]

        self.refresher = self._set_refresher()
        self.saver = self._set_saver()
        self.iterative_terminator = self._set_iterative_terminator()
        self.time_terminator = self._set_time_terminator()
        self.conditional_terminator = self._set_conditional_terminator()

        self.intersecting_curve_flag = False

        self.curr_curve = curve

    def _set_refresher(self):
        return _refresher_classes.IterativeECSFRefresher(self.refresh_interval)

    def _set_saver(self):
        return _saver_classes.IterativeECSFSaver(self.save_interval)

    def _set_iterative_terminator(self):
        return _terminator_classes.IterativeECSFTerminator(self.max_iterations)

    def _set_time_terminator(self):
        return _terminator_classes.TimeECSFTerminator(self.max_seconds)

    def _set_conditional_terminator(self):
        return _terminator_classes.ConditionalECSFTerminator(self.concavity_threshold)

    def _curr_curve_area_percent_of_original(self):
        return 100 * _metrics.enclosed_area(self.curr_curve) / self.initial_area

    def _is_time_to_resample(self):
        return True

    def _resample(self):
        self.curr_curve = _utils.resample(self.curr_curve, self.resampling_factor)

    def _filtered_resample(self):
        self.curr_curve = _utils.gaussian_filter(_utils.resample(self.curr_curve, self.resampling_factor),
                                                 self.resample_sigma)

    def _filtered_step_vector(self):
        return _utils.gaussian_filter(self._step_vector(), self.step_sigma)

    def _step_vector(self):
        return self.step_size * self._magnitude_array()[:, None] * self._vector_array()

    def _magnitude_array(self):
        # Magnitudes of iteration for each vertex.
        # Calculated as a scaled version of the normalised curvature.

        return self.scaling_function(_metrics.normalised_curvature_positive_l1(self.curr_curve))

    def _vector_array(self):
        # Vectors of iteration for each vertex.
        # Calculated as parallel to the normal and facing inward.

        return _vector_maths.inward_normal(self.curr_curve)

    def _initial_vertex_count(self):
        return self.curves[0].shape[1]

    def _initial_edge_length(self):
        return _metrics.total_edge_length(self.curves[0])

    def _initialise(self):
        self.refresher.start()
        self.saver.start()
        self.iterative_terminator.start()
        self.time_terminator.start()
        self.conditional_terminator.start(_metrics.concavity(self.curr_curve))
        self.intersecting_curve_flag = False

        self.curr_curve = self.initial_curve

        # self.curves = [self.initial_curve]
        # self.areas = [_metrics.enclosed_area(self.initial_curve)]
        self.curves = []
        self.areas = []

    def _step(self):

        curve_new = self.curr_curve + self._filtered_step_vector()

        self.curr_curve = curve_new

        if self._is_time_to_resample():
            self._filtered_resample()

        self.refresher.next_step()
        self.saver.next_step()
        self.iterative_terminator.next_step()
        self.time_terminator.next_step()
        self.conditional_terminator.next_step(_metrics.concavity(self.curr_curve))

    def run(self):
        self._initialise()

        while True:
            if (self.iterative_terminator.is_finished()
                    or self.time_terminator.is_finished()
                    or self.conditional_terminator.is_finished()):
                break

            if self.intersecting_curve_flag:
                print("Intersection in subset curve, try a smaller step size.")
                break

            if self.refresher.is_time_to_refresh():
                self.refresher.perform_refreshing(_metrics.concavity(self.curr_curve),
                                                  self._curr_curve_area_percent_of_original())

            if self.saver.is_time_to_save():
                self.areas.append(_metrics.enclosed_area(self.curr_curve))
                if len(self.areas) > 1 and self.areas[-1] > self.areas[-2]:
                    self.intersecting_curve_flag = True
                self.curves.append(self.curr_curve)

            self._step()
