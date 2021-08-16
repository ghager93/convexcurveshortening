import numpy as np

import _curveshortening_deprecated
import _image_processing
import _image_curve


def silhouette_subset_image(path: str, destination: str, original_image_path: str = None,
                            n_silhouettes: int = 10, add_border: bool = True):
    """
    Deprecated. Create image of subsets from silhouette.

    :param path: File path for smoothed silhouette.
    :param destination: File path for output image.
    :param original_image_path: File path for original silhouette image.
    :param n_silhouettes: Number of subsets in output.
    :param add_border: highlight the borders of each subset.
    :return: None.
    """
    image_curves = _image_curve_subset(path, n_silhouettes)

    silhouettes = [_fill_silhouette(image_curve) for image_curve in image_curves]

    if original_image_path is not None:
        silhouettes[0], image_curves[0] = _add_original_image(original_image_path)

    if add_border:
        borders = [_image_processing.dilate_image(image_curve) for image_curve in image_curves]
        silhouettes = [_combine_silhouette_and_border(silhouettes[i], borders[i]) for i in range(len(silhouettes))]

    try:
        image = _image_processing.int_to_rgb_colour(sum(silhouettes))

        _image_processing.save_image(image, destination)
    except ValueError:
        print('Image curves not same size.')


def silhouette_subset(path: str, n_silhouettes: int = 10, original_image_path: str = None):
    """
    Deprecated. Create subsets for image from path. Flood fill each subset.

    :param path:
    :param n_silhouettes:
    :param original_image_path:
    :return: List of filled subsets
    """
    image_curves = _image_curve_subset(path, n_silhouettes)

    if original_image_path is not None:
        _, image_curves[0] = _add_original_image(original_image_path)

    return [_fill_silhouette(image_curve) for image_curve in image_curves]


def _add_original_image(path: str):
    """
    Deprecated. Return a dilated and flood filled image and its image curve.

    :param path:
    :return:
    """
    image = _image_processing.load_image(path)
    image = np.pad(image, 10)
    curve = _image_curve.ImageCurve(image).curve()
    image_curve = _image_processing.dilate_image(_image_curve.curve_to_image_matrix(curve, image.shape))
    image = _fill_silhouette(image_curve)

    return image, image_curve


def _curve_subset(path: str, n_silhouettes: int = 10):
    """
    Deprecated. ECSF algorithm with loading and preprocessing of image.

    :param path:
    :param n_silhouettes:
    :return: list of image_curves from ECSF algorithm.
    """
    im = _image_processing.load_image(path)
    im = np.pad(im, 10)
    im_opened = _image_processing.open_image(im, 50)

    curve = _image_curve.ImageCurve(im_opened).curve()

    curves = _curveshortening_deprecated.enclosed_curve_shortening_flow(curve, n_silhouettes)
    curves[0] = _image_curve.ImageCurve(im).curve()

    return curves


def _curve_subset_preopened(path: str, unopened_path: str, n_silhouettes: int = 10):
    im = _image_processing.load_image(path)
    im = np.pad(im, 10)

    curve = _image_curve.ImageCurve(im).curve()

    curves = _curveshortening_deprecated.enclosed_curve_shortening_flow(curve, n_silhouettes)
    curves[0] = _image_curve.ImageCurve(_image_processing.load_image(unopened_path)).curve()

    return curves


def _image_curve_subset(path: str, n_silhouettes: int = 10):
    image = _image_processing.load_image(path)
    image = np.pad(image, 10)
    curves = _curve_subset(path, n_silhouettes)

    return [_image_processing.dilate_image(_image_curve.curve_to_image_matrix(curve, image.shape))
            for curve in curves]


def _fill_silhouette(image: np.ndarray):
    return _image_processing.flood_fill(image)


def _combine_silhouette_and_border(silhouette: np.ndarray, border: np.ndarray):
    try:
        output = silhouette + border / (silhouette + border).max()
    except ValueError:
        print("Silhouette and border don't have same size.")
        output = silhouette

    return output

