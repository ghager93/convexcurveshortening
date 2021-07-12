import numpy as np

from _vector_maths import tangent, normal, edge_length


def _average_radius(curve: np.ndarray):
    return np.linalg.norm(curve-curve.mean(axis=0), axis=1).mean()


def _curvature(curve: np.ndarray):
    # Curvature is the cross-product of the normal and the tangent, divided by a normalising factor.
    # k = t x n / ||t||**3
    #   = x'y" - x"y' / (x'**2 + y'**2)**(3/2)

    tangent = tangent(curve)
    normal = normal(curve)

    return ((tangent[:, 1] * normal[:, 0] - normal[:, 1] * tangent[:, 0])
            / ((tangent[:, 1] ** 2 + tangent[:, 0] ** 2) ** (3 / 2)))


def _normalise_curvature(curvature: np.ndarray):
    # Restrict curvature to between [-1, 1]
    return curvature / np.max(abs(curvature))


def _total_edge_length(curve: np.ndarray):
    return edge_length(curve).sum()


def _concavity(curve: np.ndarray):
    curvature = _curvature(curve)

    return -sum(curvature[curvature < 0])