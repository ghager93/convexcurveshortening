import os
import numpy as np

from unittest import TestCase

import _metrics
import _utils
import _vector_maths
import _curveshortening_deprecated


class Test(TestCase):
    test_dir = r"C:\Users\ghage\PycharmProjects\convexcurveshortening\lib\test_data"
    test_file_curve = os.path.join(test_dir, "heart_curve.npy")
    test_file_curve_downsample = os.path.join(test_dir, "heart_curve_downsample_100.npy")

    def test_convex_curve_shortening_flow(self):
        self.fail()

    def test__magnitude_array(self):
        self.fail()

    def test__vector_array(self):
        self.fail()

    def test__normalise_curvature(self):
        self.fail()

    def test__scale_curvature(self):
        self.fail()

    def test__concavity(self):
        self.fail()

    def test__break_condition(self):
        self.fail()

    def test__gaussian_filter(self):
        self.fail()

    def test__resample_factor_equals_1_over_100(self):
        test_input = np.load(self.test_file_curve)
        output = np.load(self.test_file_curve_downsample)

        self.assertTrue(np.allclose(_utils.resample(test_input, 1 / 100), output))

    def test__resample_factor_equals_0(self):
        test_input = np.load(self.test_file_curve)

        self.assertTrue(len(_utils.resample(test_input, 0)) == 0)

    def test__resample_returns_test_input_with_no_change(self):
        test_input = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        factor = 4 / 8

        self.assertTrue(np.allclose(_utils.resample(test_input, factor), test_input))

    def test__reduce_concave_iterations_to_precision_single_curve_circle_precision_10(self):
        test_inputx = np.cos(2 * np.pi * np.linspace(0, 1, 100, endpoint=False))
        test_inputy = np.sin(2 * np.pi * np.linspace(0, 1, 100, endpoint=False))

        test_input = [np.vstack((test_inputx, test_inputy)).transpose()]

        output = test_input

        self.assertTrue(np.allclose(_curveshortening_deprecated._reduce_concave_iterations_to_precision(test_input, 10), output))
