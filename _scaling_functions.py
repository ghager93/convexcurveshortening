import numpy as np


def f_elu(alpha, a):
    '''
    Exponential Linear Unit function constructor.  Returns an ELU function with the given hyperparameters.

    ELU(x) = { alpha * x    if  x > 0
             { a * (exp(x) - 1) otherwise

    :param alpha: Gradient of linear component of function.
    :param a: Adjusts gradient of negative domain of function.
     Setting a to non-positive will make the function non-negative.
     :return:  ELU function with hyperparameters alpha and a.
    '''

    def _elu(x):
        return np.where(x > 0, alpha * x, a * (np.exp(x) - 1))

    return _elu


def f_softplus(alpha, a):
    '''
    Softplus function constructor.  Returns a softplus function with the given hyperparameters.

    softplus(x) = a * log(1 + exp(alpha * x / a))

    :param alpha:  Gradient of linear component of function.
    :param a: Changes sharpness of the function at zero. Must be a positive number.
    :return: Softplus function with hyperparameters alpha and a.
    '''

    def _softplus(x):
        return a * np.log(1 + np.exp(alpha * x / a))

    return _softplus


def f_sigmoid(alpha, a):
    '''
    Sigmoid function constructor.  Returns a sigmoid function with the given hyperparameters.

    sigmoid(x) = 1 / (1 + np.exp(-alpha * (x - a)))

    Note: Values of x > 10 / alpha and < -10 / alpha are clipped to those boundaries.

    :param alpha: Determines the sharpness of the curve.
    :param a: Midpoint of the function's step.
    :return: Sigmoid function with hyperparameters alpha and a.
    '''

    def _sigmoid(x):
        x = np.clip(x, -10 / alpha, 10 / alpha)
        return 1 / (1 + np.exp(-alpha * (x - a)))

    return _sigmoid
