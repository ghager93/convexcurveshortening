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

    def test_edge_length_four_point_unit_length_square(self):
        test_input = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        output = np.array([1, 1, 1, 1])

        self.assertTrue(np.allclose(_vector_maths.edge_length(test_input), output))

    def test_edge_length_four_point_spiral(self):
        test_input = np.array([[0, 0], [0, 1], [1, 2], [2, -1]])
        output = np.array([np.sqrt(5), 1, np.sqrt(2), np.sqrt(10)])

        self.assertTrue(np.allclose(_vector_maths.edge_length(test_input), output))

    def test_tangent_four_point_square(self):
        test_input = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        output = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.assertTrue(np.allclose(_vector_maths.tangent(test_input), output))

    def test_normal_L_shape(self):
        test_input = np.array([[0, 0], [0, 1/2], [0, 1], [1/2, 1], [1, 1], [1, 3/2],
                          [1, 2], [3/2, 2], [2, 2], [2, 1], [2, 0], [1, 0]])
        output = np.array([[4/3, 4/3], [0, 0], [2, -2], [0, 0], [-2, 2], [0, 0], [2, -2],
                           [0, 0], [-4/3, -4/3], [0, 0], [-1, 1], [0, 0]])

        self.assertTrue(np.allclose(_vector_maths.normal(test_input), output))

    def test_inward_normal_L_shape(self):
        test_input = np.array([[0, 0], [0, 1/2], [0, 1], [1/2, 1], [1, 1], [1, 3/2],
                          [1, 2], [3/2, 2], [2, 2], [2, 1], [2, 0], [1, 0]])
        output = np.array([[0, 1], [1, 0], [1, 0], [0, -1], [0, -1], [1, 0], [1, 0],
                           [0, -1], [0, -1], [-1, 0], [-1, 0], [0, 1]])

        self.assertTrue(np.allclose(_vector_maths.inward_normal(test_input), output))
