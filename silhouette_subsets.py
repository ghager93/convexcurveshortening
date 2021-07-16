import numpy as np

import curveshortening
import _image_processing
import _image_curve


def silhouette_subset(path: str, n_silhouettes: int = 10):
    image_curves = _image_curve_subset(path, n_silhouettes)

    return [_fill_silhouette(image_curve) for image_curve in image_curves]


def _curve_subset(path: str, n_silhouettes: int = 10):
    im = _image_processing.load_image(path)
    im = np.pad(im, 10)
    im_smoothed = _image_processing.open_image(im, 50)

    curve = _image_curve.ImageCurve(im_smoothed).curve()

    curves = curveshortening.enclosed_curve_shortening_flow(curve, n_silhouettes)
    curves[0] = _image_curve.ImageCurve(im).curve()

    return curves


def _image_curve_subset(path: str, n_silhouettes: int = 10):
    image = _image_processing.load_image(path)
    curves = _curve_subset(path, n_silhouettes)

    return [_image_processing.dilate_image(_image_curve.curve_to_image_matrix(curve, image.shape))
            for curve in curves]


def _fill_silhouette(image: np.ndarray):
    dilated_image = _image_processing.dilate_image(image)
    return _image_processing.flood_fill(dilated_image)


def silhouette_subset_image(path: str, destination: str, n_silhouettes: int = 10, add_border: bool = True):
    image_curves = _image_curve_subset(path, n_silhouettes)

    silhouettes = [_fill_silhouette(image_curve) for image_curve in image_curves]

    if add_border:
        borders = [_image_processing.dilate_image(image_curve) for image_curve in image_curves]
        silhouettes = [combine_silhouette_and_border(silhouettes[i], borders[i]) for i in range(silhouettes)]

    image = _image_processing.int_to_rgb_colour(sum(silhouettes))

    _image_processing.save_image(image, destination)


def combine_silhouette_and_border(silhouette: np.ndarray, border: np.ndarray):
    try:
        output = silhouette + border / (silhouette + border).max()
    except ValueError:
        print('silhouette and border should be same dimensions')
        output = silhouette

    return output

