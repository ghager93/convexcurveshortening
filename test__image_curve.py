import numpy as np

from unittest import TestCase
from _image_curve import _edge_detect, ImageCurve


class Test(TestCase):
    def test__edge_detect_3_by_3_low(self):
        input = np.zeros((3, 3), dtype=int)
        output = np.zeros((3, 3), dtype=int)

        self.assertTrue(np.allclose(_edge_detect(input), output))

    def test__edge_detect_3_by_3_high(self):
        input = np.ones((3, 3), dtype=int)
        output = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        self.assertTrue(np.allclose(_edge_detect(input), output))

    def test_imagecurve_no_change_im(self):
        input = np.array([[0, 0],
                          [0, 1]])
        output = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]])

        self.assertTrue(np.allclose(ImageCurve(input)._im, output))

    def test_imagecurve_prepare_im_invert(self):
        input = np.array([[1, 0],
                          [0, 1]])
        output = np.array([[0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0]])

        self.assertTrue(np.allclose(ImageCurve(input)._im, output))

    def test_imagecurve_prepare_im_flatten(self):
        input = np.array([[0, 2],
                          [0, 0]])
        output = np.array([[0, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])

        self.assertTrue(np.allclose(ImageCurve(input)._im, output))

    def test_imagecurve_curve_3_by_3_low(self):
        input = np.zeros((3, 3), dtype=int)
        output = np.array([])

        self.assertTrue(np.allclose(ImageCurve(input).curve(), output))

    def test_imagecurve_starting_point(self):
        input = np.zeros((5, 5), dtype=int)
        input[1:-1, 1:-1] = 1
        pad = (1, 1)

        output = np.array([1, 1]) + pad

        self.assertTrue(np.allclose(ImageCurve(input)._starting_point(), output))

    def test_imagecurve_second_point(self):
        input = np.zeros((5, 5), dtype=int)
        input[1:-1, 1:-1] = 1
        pad = (1, 1)

        output = np.array([1, 2]) + pad

        self.assertTrue(np.allclose(ImageCurve(input)._second_point(), output))

    def test_imagecurve_last_point(self):
        input = np.zeros((5, 5), dtype=int)
        input[1:-1, 1:-1] = 1
        pad = (1, 1)

        output = np.array([2, 1]) + pad

        self.assertTrue(np.allclose(ImageCurve(input)._last_point(), output))

    def test_imagecurve_curve_3_by_3_high(self):
        input = np.zeros((5, 5), dtype=int)
        input[1:-1, 1:-1] = 1

        output = np.array([[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1]])

        self.assertTrue(np.allclose(ImageCurve(input).curve(), output))
