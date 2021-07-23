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
