import numpy as np


def tangent(curve: np.ndarray):
    edge = edge_length(curve)
    edge[edge == 0] = 10e-5

    return (curve - np.roll(curve, 1, axis=0)) / edge[:, None]


def normal(curve: np.ndarray):
    tangent = tangent(curve)
    edge = edge_length(curve)
    second_diff_edge = np.roll(edge, -1, axis=0) + edge
    second_diff_edge[second_diff_edge == 0] = 10e-5

    return 2 * (np.roll(tangent, -1, axis=0) - tangent) / second_diff_edge[:, None]


def inward_normal(curve: np.ndarray):
    # The inward normal is the normal pointing toward the interior of a closed curve.
    # It is the tangent vector rotated clockwise 90 degrees.

    return tangent(curve) @ np.array([[0, -1], [1, 0]])


def edge_length(curve: np.ndarray):
    # Euclidean distance between each neighbouring vertex.

    return np.linalg.norm(curve - np.roll(curve, 1, axis=0), axis=1)