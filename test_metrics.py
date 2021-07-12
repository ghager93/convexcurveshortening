import os
import numpy as np

from unittest import TestCase

import _metrics
import _vector_maths
import curveshortening


class Test(TestCase):
    test_dir = r"C:\Users\ghage\PycharmProjects\convexcurveshortening\lib\test_data"
    test_file_curve = os.path.join(test_dir, "heart_curve.npy")
    test_file_curve_downsample = os.path.join(test_dir, "heart_curve_downsample_100.npy")

    def test__curvature_parabola_y_equals_x_squared(self):
        inputx = np.linspace(-4, 4, 100)
        inputy = inputx**2

        input = np.vstack((inputy, inputx)).transpose()

        output = 2 / (4*inputx**2 + 1)**(3/2)

        self.assertTrue(np.allclose(_metrics.curvature(input)[1:-1], output[1:-1], atol=1.e-3, rtol=1.e-1))

    def test_curvature_straight_line(self):
        input = np.array([np.arange(10), np.arange(10)]).transpose()

        output = np.zeros(10)

        self.assertTrue(np.allclose(_metrics.curvature(input)[1:-1], output[1:-1]))

    def test_normalised_curvature(self):
        self.fail()

    def test__concavity(self):
        self.fail()

    def test_area_unit_square(self):
        input = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

        output = 1

        self.assertEqual(_metrics.enclosed_area(input), output)

    def test_area_oval_1000_points(self):
        inputx = 100 * np.cos(np.linspace(0, 2*np.pi, 1000, endpoint=False))
        inputy = 20 * np.sin(np.linspace(0, 2*np.pi, 1000, endpoint=False))

        input = np.vstack((inputx, inputy)).transpose()

        output = 100 * 20 * np.pi

        self.assertTrue(np.isclose(_metrics.enclosed_area(input), output))
