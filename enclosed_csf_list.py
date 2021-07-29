import numpy as np
import skgeom

from typing import List

import _concave_enclosed_csf_list
import _csf_list
import _image_curve


def enclosed_csf_list(curve: np.ndarray, n_subsets: int, step_size: float = 1):
    """
    Runs the enclosed curve shortening flow algorithm on a curve and returns n_subsets curves that are have an enclosed
    area linearly-spaced between the initial curves enclosed area and zero.

    :param curve: Nx2 Numpy array of the outline of an image.  Curve must run clockwise.
    :param n_subsets: Number of subsets to return. Subsets are linearly proportional of the area of the initial outline.
    :param step_size: Step size for each iteration of the concave ECSF algorithm.
    If the algorithm fails, try a smaller step size.
    :return:  n_subsets long list of 2D numpy arrays.
    """
    ecsf_obj = _concave_enclosed_csf_list.ConcaveEnclosedCSFList(curve, step_size=step_size)

    try:
        ecsf_obj.run()
    except Exception as curve_loop_detected:
        raise curve_loop_detected

    num_concave_curves = max(1, int((1 - ecsf_obj.last_to_first_curve_area_ratio()) * n_subsets))

    concave_curves = ecsf_obj.get_n_curves(num_concave_curves)

    csf_obj = _csf_list.CSFList(concave_curves[-1])

    convex_curves = csf_obj.mm_subset(n_subsets - num_concave_curves + 1)

    return concave_curves + convex_curves[1:]


def enclosed_csf_list_retry_on_fail(curve: np.ndarray, n_subsets: int, step_size: float = 1):
    """
    Runs enclosed_csf_list(). If algorithm fails, step_size is reduced by a factor of 5 and the algorithm is run again.
    Fails if algorithm fails 4 times.

    :param curve: Nx2 Numpy array of the outline of an image.  Curve must run clockwise.
    :param n_subsets: Number of subsets to return. Subsets are linearly proportional of the area of the initial outline.
    :param step_size: Step size for each iteration of the concave ECSF algorithm.
    If the algorithm fails, try a smaller step size.
    :return:  n_subsets long list of 2D numpy arrays.
    """
    for attempt in range(4):
        try:
            ecsf_list =  enclosed_csf_list(curve, n_subsets, step_size)
        except:
            step_size /= 5
        else:
            break
    else:
        raise Exception(f"Loop detected with step size {step_size}. Curve cannot be shortened.")

    return ecsf_list


def to_image_matrix(ecsf_list: List):
    polygon = skgeom.Polygon(ecsf_list[0])
    bbox = polygon.bbox()
    xmin, xmax = bbox.xmin(), bbox.xmax()
    ymin, ymax = bbox.ymin(), bbox.ymax()
    pad = 10
    shape = int(xmax - xmin + 2*pad), int(ymax - ymin + 2*pad)

    out = np.zeros(shape, dtype=float)
    for i, curve in enumerate(ecsf_list):
        try:
            out = _image_curve.imprint_curve_on_matrix(curve, out, i+1)
        except:
            continue

    return out