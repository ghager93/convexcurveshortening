import numpy as np

from unittest import TestCase

import _image_processing


class Test(TestCase):
    def test_flood_fill_square(self):
        input = np.zeros((7, 7))
        input[1:-1, 1:-1] = 1
        input[2:-2, 2:-2] = 0

        output = np.zeros((7, 7))
        output[1:-1, 1:-1] = 1

        self.assertTrue(np.allclose(_image_processing.flood_fill(input), output))

    def test_flood_fill_centre_not_in_curve(self):
        input = np.zeros((16, 16))
        input[1:-1, 1:-1] = 1
        input[2:-2, 2:-2] = 0
        input[4:-1, 4:-4] = 1
        input[5:-1, 5:-5] = 0

        output = np.zeros((16, 16))
        output[1:-1, 1:-1] = 1
        output[5:-1, 5:-5] = 0

        self.assertTrue(np.allclose(_image_processing.flood_fill(input), output))

    def test_flood_fill_curve_too_small_to_fill(self):
        input = np.zeros((6, 6))
        input[2:3, 2:3] = 1

        output = input

        self.assertTrue(np.allclose(_image_processing.flood_fill(input), output))

    def test_flood_fill_curve_around_border(self):
        input = np.ones((5, 5))
        input[1:-1, 1:-1] = 0

        output = np.ones((5, 5))

        self.assertTrue(np.allclose(_image_processing.flood_fill(input), output))


    def test_flood_fill_cv_square(self):
        input = np.zeros((7, 7))
        input[1:-1, 1:-1] = 1
        input[2:-2, 2:-2] = 0

        output = np.zeros((7, 7))
        output[1:-1, 1:-1] = 1

        self.assertTrue(np.allclose(_image_processing.flood_fill_cv(input), output))

    def test_flood_fill_cv_centre_not_in_curve(self):
        input = np.zeros((16, 16))
        input[1:-1, 1:-1] = 1
        input[2:-2, 2:-2] = 0
        input[4:-1, 4:-4] = 1
        input[5:-1, 5:-5] = 0

        output = np.zeros((16, 16))
        output[1:-1, 1:-1] = 1
        output[5:-1, 5:-5] = 0

        self.assertTrue(np.allclose(_image_processing.flood_fill_cv(input), output))

    def test_flood_fill_cv_curve_too_small_to_fill(self):
        input = np.zeros((6, 6))
        input[2:3, 2:3] = 1

        output = input

        self.assertTrue(np.allclose(_image_processing.flood_fill_cv(input), output))

    def test_flood_fill_cv_curve_around_border(self):
        input = np.ones((5, 5))
        input[1:-1, 1:-1] = 0

        output = np.ones((3, 3))
        output = np.pad(output, (1, 1))

        self.assertTrue(np.allclose(_image_processing.flood_fill_cv(input), output))
