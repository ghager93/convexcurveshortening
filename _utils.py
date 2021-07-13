import numpy as np
from scipy import ndimage, interpolate

import _vector_maths


def _gaussian_filter(curve: np.ndarray, sigma: float):
    return ndimage.gaussian_filter1d(curve, sigma, axis=0, mode='wrap')


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