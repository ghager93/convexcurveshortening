import numpy as np

from unittest import TestCase

import _image_processing


class Test(TestCase):
    def test_flood_fill(self):
        input = np.zeros((7, 7))
        input[1:-1, 1:-1] = 1
        input[2:-2, 2:-2] = 0

        output = np.zeros((7, 7))
        output[1:-1, 1:-1] = 1

        self.assertTrue(np.allclose(_image_processing.flood_fill(input), output))
