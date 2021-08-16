import os
import numpy as np

from unittest import TestCase

import _metrics
import _vector_maths
import _curveshortening_deprecated


class Test(TestCase):
    test_dir = r"C:\Users\ghage\PycharmProjects\convexcurveshortening\lib\test_data"
    test_file_curve = os.path.join(test_dir, "heart_curve.npy")
    test_file_curve_downsample = os.path.join(test_dir, "heart_curve_downsample_100.npy")

    def test__curvature_parabola_y_equals_x_squared(self):
        test_inputx = np.linspace(-4, 4, 100)
        test_inputy = test_inputx**2

        test_input = np.vstack((test_inputy, test_inputx)).transpose()

        output = 2 / (4*test_inputx**2 + 1)**(3/2)

        self.assertTrue(np.allclose(_metrics.curvature(test_input)[1:-1], output[1:-1], atol=1.e-3, rtol=1.e-1))

    def test_curvature_straight_line(self):
        test_input = np.array([np.arange(10), np.arange(10)]).transpose()

        output = np.zeros(10)

        self.assertTrue(np.allclose(_metrics.curvature(test_input)[1:-1], output[1:-1]))

    def test_normalised_curvature(self):
        self.fail()

    def test__concavity(self):
        self.fail()

    def test_area_unit_square(self):
        test_input = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

        output = 1

        self.assertEqual(_metrics.enclosed_area(test_input), output)

    def test_area_oval_1000_points(self):
        test_inputx = 100 * np.cos(np.linspace(0, 2*np.pi, 1000, endpoint=False))
        test_inputy = 20 * np.sin(np.linspace(0, 2*np.pi, 1000, endpoint=False))

        test_input = np.vstack((test_inputx, test_inputy)).transpose()

        output = 100 * 20 * np.pi

        self.assertTrue(np.isclose(_metrics.enclosed_area(test_input), output))
