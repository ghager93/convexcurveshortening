import numpy as np


def convex_curve_shortening_flow(curve):
    '''
    Convex curve shortening.

    :param curve:
    :return:
    '''

    if type(curve) is not np.ndarray:
        raise ValueError('Curve must be 2-D Numpy array')
