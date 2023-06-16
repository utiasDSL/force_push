import pytest
import numpy as np

import force_push as fp


def test_line_segment_to_point_dist_x():
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    p = np.array([0.5, 0.5])

    d, c = fp.line_segment_to_point_dist(v1, v2, p)

    assert np.isclose(d, 0.5)
    assert np.allclose(c, [0.5, 0])


def test_line_segment_to_point_dist_diag():
    v1 = np.array([-1, 0])
    v2 = np.array([-2, 1])
    p = np.array([-1, 1])

    d, c = fp.line_segment_to_point_dist(v1, v2, p)

    assert np.isclose(d, np.sqrt(2) / 2)
    assert np.allclose(c, [-1.5, 0.5])


def test_line_segment_to_point_dist_point():
    # the line is actually a point
    v1 = np.array([1, 0])
    v2 = np.array([1, 0])
    p = np.array([0.5, 0.5])

    d, c = fp.line_segment_to_point_dist(v1, v2, p)

    assert np.isclose(d, np.sqrt(2) / 2)
    assert np.allclose(c, v1)


def test_line_segment_to_point_dist_on_line():
    # the point is on the line
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    p = np.array([0.5, 0])

    d, c = fp.line_segment_to_point_dist(v1, v2, p)

    assert np.isclose(d, 0)
    assert np.allclose(c, p)


def test_line_segment_to_point_dist_beyond_end():
    # the point is beyond the end of the line segment
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    p = np.array([2, 0])

    d, c = fp.line_segment_to_point_dist(v1, v2, p)

    assert np.isclose(d, 1)
    assert np.allclose(c, v2)


def test_segment_path():
    vertices = np.array([[0, 0], [1, 0]])
    path = fp.SegmentPath(vertices, final_direction=[0, 1])

    p = np.array([2, 1])
    c = path.compute_closest_point(p)
    direction, offset = path.compute_direction_and_offset(p)

    assert np.allclose(offset, -1)
    assert np.allclose(c, [1, 1])
    assert np.allclose(direction, [0, 1])


def test_square_path():
    path = fp.SegmentPath.square(half_length=1.0, center=[1, 1])

    p = np.array([3, 1])
    c = path.compute_closest_point(p)
    direction, offset = path.compute_direction_and_offset(p)

    assert np.allclose(offset, -1)
    assert np.allclose(c, [2, 1])
    assert np.allclose(direction, [0, 1])


def test_line_path():
    path = fp.SegmentPath.line([1, 0])

    p = np.array([1, 1])
    c = path.compute_closest_point(p)
    direction, offset = path.compute_direction_and_offset(p)

    assert np.allclose(offset, 1)
    assert np.allclose(c, [1, 0])
    assert np.allclose(direction, [1, 0])
