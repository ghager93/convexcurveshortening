import os
import numpy as np

from unittest import TestCase

import curveshortening


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

    def test__curvature_parabola_y_equals_x_squared(self):
        inputx = np.linspace(-4, 4, 100)
        inputy = inputx**2

        input = np.vstack((inputy, inputx)).transpose()

        output = 2 / (4*inputx**2 + 1)**(3/2)

        self.assertTrue(np.allclose(curveshortening._curvature(input)[1:-1], output[1:-1], atol=1.e-3, rtol=1.e-1))

    def test__normalise_curvature(self):
        self.fail()

    def test__scale_curvature(self):
        self.fail()

    def test__edge_length_four_point_unit_length_square(self):
        input = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        output = np.array([1, 1, 1, 1])

        self.assertTrue(np.allclose(curveshortening._edge_length(input), output))

    def test__edge_length_four_point_spiral(self):
        input = np.array([[0, 0], [0, 1], [1, 2], [2, -1]])
        output = np.array([np.sqrt(5), 1, np.sqrt(2), np.sqrt(10)])

        self.assertTrue(np.allclose(curveshortening._edge_length(input), output))

    def test__tangent_four_point_square(self):
        input = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        output = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.assertTrue(np.allclose(curveshortening._tangent(input), output))

    def test__normal_L_shape(self):
        input = np.array([[0, 0], [0, 1/2], [0, 1], [1/2, 1], [1, 1], [1, 3/2],
                          [1, 2], [3/2, 2], [2, 2], [2, 1], [2, 0], [1, 0]])
        output = np.array([[4/3, 4/3], [0, 0], [2, -2], [0, 0], [-2, 2], [0, 0], [2, -2],
                           [0, 0], [-4/3, -4/3], [0, 0], [-1, 1], [0, 0]])

        self.assertTrue(np.allclose(curveshortening._normal(input), output))

    def test__inward_normal_L_shape(self):
        input = np.array([[0, 0], [0, 1/2], [0, 1], [1/2, 1], [1, 1], [1, 3/2],
                          [1, 2], [3/2, 2], [2, 2], [2, 1], [2, 0], [1, 0]])
        output = np.array([[0, 1], [1, 0], [1, 0], [0, -1], [0, -1], [1, 0], [1, 0],
                           [0, -1], [0, -1], [-1, 0], [-1, 0], [0, 1]])

        self.assertTrue(np.allclose(curveshortening._inward_normal(input), output))

    def test__concavity(self):
        self.fail()

    def test__break_condition(self):
        self.fail()

    def test__gaussian_filter(self):
        self.fail()

    def test__resample_factor_equals_1_over_100(self):
        input = np.load(self.test_file_curve)
        output = np.load(self.test_file_curve_downsample)

        self.assertTrue(np.allclose(curveshortening._resample(input, 1/100), output))

    def test__resample_factor_equals_0(self):
        input = np.load(self.test_file_curve)

        self.assertTrue(len(curveshortening._resample(input, 0)) == 0)

    def test__resample_returns_input_with_no_change(self):
        input = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        factor = 4 / 8

        self.assertTrue(np.allclose(curveshortening._resample(input, factor), input))
