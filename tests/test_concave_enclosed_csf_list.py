from unittest import TestCase

import numpy as np

from _concave_enclosed_csf_list import ConcaveEnclosedCSFList


class TestConcaveEnclosedCSFList(TestCase):
    def test_filtered_step_vector_circular_curve_result_points_inward(self):
        test_input_curve = np.vstack((500 * (np.cos(np.linspace(0, 2*np.pi)) + 1.2)),
                                (500 * (np.sin(np.linspace(0, 2*np.pi)) + 1.2)))
        ConcaveEnclosedCSFList(test_input_curve)


