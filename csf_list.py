import numpy as np

import _metrics
import _vector_maths
import _utils


class CSFList:
    """
    Curve shortening flow.

    :param curve: Nx2 Numpy array, where N is the number of vertices in the curve.
    :param n_curves: Number of curve iterations between the original and singularity.
    :return:
    """
    def __init__(self, curve: np.ndarray):

        curve = curve.astype(float)

        self.curve = curve

        self.resampling_factor = curve.shape[0] / _metrics.total_edge_length(curve)
        self.initial_area = _metrics.enclosed_area(curve)
        self.centre_of_mass = _vector_maths.centre_of_mass(curve)
        self.centre_vectors = self.centre_of_mass - curve

    def proportion_subset(self, proportion: float):
        return self.curve + proportion * self.centre_vectors

    def linspace_subsets(self, n_subsets: int):
        return [self.proportion_subset(proportion) for proportion in np.linspace(0, 1, n_subsets)]

    def linspace_subsets_resample(self, n_subset: int):
        return [_utils.resample(curve, self.resampling_factor) for curve in self.linspace_subsets(n_subset)]

    def mm_subset(self, n_subsets: int):
        linear_stds = self._linear_step_sigmas(n_subsets)

        return [self._mokhtarian_mackworth92(sigma) for sigma in linear_stds]

    def _linear_step_sigmas(self, n_curves: int, startpoint=True):
        # Array of n_curve stds that will create linearly spaced curves.
        # Curve shortening flow algorithm found to closely match scaled normal distribution;
        # N(x; \sigma) = n_vertices \exp(-x^2 / 2\sigma^2),    \sigma = pi/20, x > 0

        sigma2 = (np.pi / 20) ** 2
        average_radius = _metrics.mean_distance_to_centre_of_mass(self.curve)

        if startpoint:
            linear_steps = np.linspace(average_radius, 0, n_curves, endpoint=False)
        else:
            linear_steps = np.linspace(average_radius, 0, n_curves + 1, endpoint=False)[1:]
        return self.curve.shape[0] * np.sqrt(2 * sigma2 * np.log(average_radius / linear_steps))

    def _mokhtarian_mackworth92(self, sigma):
        # Mokhtarian, Farzin & Mackworth, Alan. (1992).
        # A Theory of Multiscale, Curvature-Based Shape Representation for Planar Curves.
        # Pattern Analysis and Machine Intelligence, IEEE Transactions on. 14. 789-805. 10.1109/34.149591.

        if sigma == 0:
            mm_curve = self.curve
        else:
            mm_curve = _utils.gaussian_filter(self.curve, sigma)

        return mm_curve
