import numpy as np

import curveshortening
import _image_processing
import _image_curve


def silhouette_subsets(path: str, n_silhouettes: int = 10):
    im = _image_processing.load_image(path)
    im = np.pad(im, 10)
    im_smoothed = _image_processing.smooth_image(im, 50)

    curve = _image_curve.ImageCurve(im_smoothed).curve()

    curves = curveshortening.enclosed_curve_shortening_flow(curve, n_silhouettes)
    curves[0] = _image_curve.ImageCurve(im).curve()

    return [_image_curve.curve_to_image_matrix_filled(_image_curve.crop(curve, im.shape), im.shape)
            for curve in curves]


def silhouette_subsets_image(path: str, destination: str, n_silhouettes: int = 10):
    _image_processing.save_image(sum(silhouette_subsets(path, n_silhouettes)), destination)
