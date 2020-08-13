import numpy as np


def rms(e):
    ''' Calculate root mean square of a vector of data. '''
    return np.sqrt(np.mean(np.square(e)))


def bound_array(a, lb, ub):
    ''' Elementwise bound array above and below. '''
    return np.minimum(np.maximum(a, lb), ub)


def right_pseudoinverse(J):
    JJT = J.dot(J.T)
    return J.T.dot(np.linalg.inv(JJT))
