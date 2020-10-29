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


def rotation_matrix(θ):
    ''' 2D rotation matrix: rotates points counter-clockwise. '''
    return np.array([[np.cos(θ), -np.sin(θ)],
                     [np.sin(θ),  np.cos(θ)]])


def rotation_jacobian(θ):
    ''' Derivative of rotation matrix (above) w.r.t θ. '''
    return np.array([[-np.sin(θ), -np.cos(θ)],
                     [ np.cos(θ), -np.sin(θ)]])


def dist_to_line_segment(p, p1, p2):
    ''' Calculate distance and closest point of p to the line segment with end
        points p1 and p2. '''
    v = (p2 - p1)
    length2 = np.dot(v, v)
    P = (np.eye(2) - np.outer(v, v) / length2)
    a = p1 - p

    # closest point
    pc = np.dot(P, a) + p

    # squared distance between closest point and end points of the line
    d1_2 = np.dot(p1 - pc, p1 - pc)
    d2_2 = np.dot(p2 - pc, p2 - pc)

    # check if closest point falls outside the line segment
    if max(d1_2, d2_2) > length2:
        if d1_2 >= d2_2:
            pc = p2
        else:
            pc = p1

    d2 = np.dot(p - pc, p - pc)
    return pc, d2
