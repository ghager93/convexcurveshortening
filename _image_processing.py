import PIL
import numpy as np
import os
import warnings

from PIL import Image
from scipy.ndimage import morphology
from skimage import morphology

import _structuring_element


def load_image(filename: str):
    im = None

    try:
        im = np.array(Image.open(filename).convert("1"))

        # Top-left corner is assumed to be background. Therefore, if this pixel is high, the image is inverted.
        im = im ^ im[0, 0]

    except FileNotFoundError:
        print("File does not exist.")
    except PIL.UnidentifiedImageError:
        print("Image format not supported.")
    except:
        print("Something went wrong.")

    return im


def smooth_image(image: np.ndarray, factor: int = 1):
    if image.ndim != 2:
        raise ValueError('Image must be 2D.')
    if factor <= 0:
        raise ValueError('Smoothing factor must be positive integer.')
    return morphology.binary_closing(morphology.binary_opening(image, _structuring_element.circle(factor).kernel),
                                     _structuring_element.circle(factor).kernel)


def flood_fill(image: np.ndarray):
    # Fill in pixels of binary image of a closed curve.
    # The method will first assume the middle pixel is inside the curve, and hence fill from here.
    # If the result filled the top-most pixel, the fill was of the outside of the curve.
    # The method will then run the method again on the opposite pixels.
    # Assumes there is one closed curve that is simple.

    seed_pixel = int(image.shape[0]//2), int(image.shape[1]//2)

    # Check if seed pixel is part of curve
    if image[seed_pixel]:
        seed_pixel = np.argwhere(~image[seed_pixel[0]:, seed_pixel[1]:])[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filled_im = morphology.flood_fill(image, seed_pixel, 1)

    if filled_im[0, 0]:
        seed_pixel = np.argwhere(~filled_im)[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filled_im = morphology.flood_fill(image, seed_pixel, 1)

    return filled_im

