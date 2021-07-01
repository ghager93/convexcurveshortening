import PIL
import numpy as np
import os

from PIL import Image
from scipy.ndimage import morphology

import _structuring_element


def load_image(filename: str):
    try:
        im = Image.open(filename).convert("1")
    except FileNotFoundError:
        print("File does not exist.")
    except PIL.UnidentifiedImageError:
        print("Image format not supported.")


def smooth_image(image: np.ndarray, factor: int = 1):
    if image.ndim != 2:
        raise ValueError('Image must be 2D.')
    if factor <= 0:
        raise ValueError('Smoothing factor must be positive integer.')
    return morphology.binary_opening(image, _structuring_element.circle(factor).kernel)
