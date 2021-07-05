import PIL
import numpy as np
import os

from PIL import Image
from scipy.ndimage import morphology

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
