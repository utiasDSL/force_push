import pytest
import numpy as np
from spatialmath.base import rotz

import force_push as fp


def test_line_segment_to_point_dist_x():
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    p = np.array([0.5, 0.5])

    segment = fp.LineSegment(v1, v2)
    info = segment.closest_point_info(p)

    assert np.isclose(info.deviation, 0.5)
    assert np.allclose(info.point, [0.5, 0])
    assert np.allclose(info.direction, segment.direction)
    assert np.isclose(info.distance_from_start, 0.5)


def test_line_segment_to_point_dist_diag():
    v1 = np.array([-1, 0])
    v2 = np.array([-2, 1])
    p = np.array([-1, 1])

    segment = fp.LineSegment(v1, v2)
    info = segment.closest_point_info(p)

    assert np.isclose(info.deviation, np.sqrt(2) / 2)
    assert np.allclose(info.point, [-1.5, 0.5])
    assert np.allclose(info.direction, segment.direction)
    assert np.isclose(info.distance_from_start, np.sqrt(2) / 2)


def test_line_segment_to_point_dist_point():
    # the line is actually a point
    v1 = np.array([1, 0])
    v2 = np.array([1, 0])
    p = np.array([0.5, 0.5])

    segment = fp.LineSegment(v1, v2)
    info = segment.closest_point_info(p)

    assert np.isclose(info.deviation, np.sqrt(2) / 2)
    assert np.allclose(info.point, v1)
    assert np.allclose(info.direction, segment.direction)
    assert np.isclose(info.distance_from_start, 0)


def test_line_segment_to_point_dist_on_line():
    # the point is on the line
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    p = np.array([0.5, 0])

    segment = fp.LineSegment(v1, v2)
    info = segment.closest_point_info(p)

    assert np.isclose(info.deviation, 0)
    assert np.allclose(info.point, p)
    assert np.allclose(info.direction, segment.direction)
    assert np.isclose(info.distance_from_start, 0.5)


def test_line_segment_to_point_dist_beyond_end():
    # the point is beyond the end of the line segment
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    p = np.array([2, 0])

    info = fp.LineSegment(v1, v2).closest_point_info(p)

    assert np.isclose(info.deviation, 1)
    assert np.allclose(info.point, v2)
    assert np.isclose(info.distance_from_start, 1)


def test_segment_path_lines():
    path = fp.SegmentPath(
        [fp.LineSegment([0, 0], [1, 0]), fp.LineSegment([1, 0], [1, 1], infinite=True)]
    )

    p = np.array([2, 1])
    direction, offset = path.compute_direction_and_offset(p)

    assert np.allclose(offset, -1)
    assert np.allclose(direction, [0, 1])

    info = path.compute_closest_point_info(p)
    assert np.allclose(info.deviation, 1.0)
    assert np.allclose(info.direction, [0, 1])
    assert np.allclose(info.point, [1, 1])
    assert np.allclose(info.distance_from_start, 2.0)


def test_line_path():
    path = fp.SegmentPath.line([1, 0])

    p = np.array([1, 1])
    direction, offset = path.compute_direction_and_offset(p)

    assert np.allclose(offset, 1)
    assert np.allclose(direction, [1, 0])

    info = path.compute_closest_point_info(p)
    assert np.allclose(info.deviation, 1.0)
    assert np.allclose(info.direction, [1, 0])
    assert np.allclose(info.point, [1, 0])
    assert np.allclose(info.distance_from_start, 1.0)


def test_segment_path_lines_and_arcs():
    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [3.0, 0]),
            fp.CircularArcSegment(center=[3.0, 2.0], point=[3.0, 0], angle=np.pi / 2),
            fp.LineSegment([5.0, 2], [5.0, 5], infinite=True),
        ],
    )

    # partway along the path (closest to the arc)
    p = np.array([5, 0])

    direction, offset = path.compute_direction_and_offset(p)
    assert np.allclose(offset, -2 * (np.sqrt(2) - 1))
    assert np.allclose(direction, [np.sqrt(2) / 2, np.sqrt(2) / 2])

    info = path.compute_closest_point_info(p)
    assert np.allclose(info.deviation, abs(offset))
    assert np.allclose(info.direction, direction)
    C = rotz(-np.pi / 4)[:2, :2]
    assert np.allclose(info.point, np.array([3, 2]) + C @ [2, 0])
    assert np.allclose(info.distance_from_start, 3 + np.pi / 2)

    # farther along the path (closest to the second line segment)
    p = np.array([4, 4])

    direction, offset = path.compute_direction_and_offset(p)
    assert np.allclose(offset, 1.0)
    assert np.allclose(direction, [0, 1])

    info = path.compute_closest_point_info(p)
    assert np.allclose(info.deviation, 1.0)
    assert np.allclose(info.direction, direction)
    assert np.allclose(info.point, [5, 4])
    assert np.allclose(info.distance_from_start, 3 + np.pi + 2)


# def test_bezier_curve_length():
#     v1 = [0, 0]
#     v2 = [1, 0]
#     v3 = [1, 1]
#     segment = fp.QuadBezierSegment(v1, v2, v3)
#
#     # manually approximate the path as a large number of short straight lines
#     n = 1000
#     ts = np.linspace(0, 1, n)
#     L = 0
#     for i in range(n - 1):
#         p0 = segment._evaluate(ts[i])
#         p1 = segment._evaluate(ts[i + 1])
#         L += np.linalg.norm(p1 - p0)
#
#     assert np.isclose(segment.length, L)
#
#
# def test_bezier_point_at_distance():
#     v1 = [0, 0]
#     v2 = [1, 0]
#     v3 = [1, 1]
#     segment = fp.QuadBezierSegment(v1, v2, v3)
#
#     p = segment.point_at_distance(segment.length / 2)
#     assert np.allclose(p, segment._evaluate(0.5))
#
#     with pytest.raises(AssertionError):
#         segment.point_at_distance(-0.1)
#
#     with pytest.raises(ValueError):
#         segment.point_at_distance(segment.length + 0.1)


def test_line_segment_point_at_distance():
    v1 = [0, 0]
    v2 = [1, 0]
    segment = fp.LineSegment(v1, v2, infinite=True)

    d = 10
    p = segment.point_at_distance(d)
    assert np.allclose(p, [d, 0])

    with pytest.raises(AssertionError):
        segment.point_at_distance(-0.1)


def test_segment_path_point_at_distance():
    d = 10

    # path = fp.SegmentPath(
    #     [
    #         fp.LineSegment([0.0, 0], [3.0, 0]),
    #         fp.QuadBezierSegment([3.0, 0], [5.0, 0], [5, 2]),
    #         fp.LineSegment([5.0, 2], [5.0, 5], infinite=True),
    #     ],
    # )
    #
    # # compute the path knowing the point is somewhere on the last segment
    # p = path.point_at_distance(d)
    # L = path.segments[0].length + path.segments[1].length
    # r = path.segments[-1].point_at_distance(d - L)
    # assert np.allclose(p, r)
    #
    # with pytest.raises(AssertionError):
    #     path.point_at_distance(-0.1)

    # again with a circle arc path
    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [3.0, 0]),
            fp.CircularArcSegment(center=[3.0, 3.0], point=[3.0, 0], angle=np.pi / 2),
            fp.LineSegment([5.0, 2], [5.0, 5], infinite=True),
        ],
    )
    p = path.point_at_distance(d)
    L = path.segments[0].length + path.segments[1].length
    r = path.segments[-1].point_at_distance(d - L)
    assert np.allclose(p, r)


def test_circular_arc():
    center = [1, 1]
    v1 = [1, 0]
    angle = np.pi / 2
    segment = fp.CircularArcSegment(center=center, point=v1, angle=angle)
    assert np.allclose(segment.v2, [2, 1])
    assert np.isclose(segment.length, angle)
    assert np.isclose(segment.radius, 1.0)


def test_circular_arc_closest_point_info():
    center = np.array([1, 1])
    v1 = center + [0, -1]
    angle = np.pi
    segment = fp.CircularArcSegment(center=center, point=v1, angle=angle)

    # outside the circle
    p = center + [2, 0]
    info = segment.closest_point_info(p)
    assert np.allclose(info.point, center + [1, 0])
    assert np.allclose(info.direction, [0, 1])
    assert np.isclose(info.deviation, 1.0)

    # inside the circle
    p = center + [0.5, 0]
    info = segment.closest_point_info(p)
    assert np.allclose(info.point, center + [1, 0])
    assert np.allclose(info.direction, [0, 1])
    assert np.isclose(info.deviation, 0.5)

    # in the part of the circle not surrounded by the arc
    p = center + [-0.1, 0.001]
    info = segment.closest_point_info(p)
    assert np.allclose(info.point, center + [0, 1])
    assert np.allclose(info.direction, [-1, 0])

    # exactly at the center
    # the point and direction are an implementation detail, but the deviation
    # is known
    info = segment.closest_point_info(center)
    assert np.isclose(info.deviation, segment.radius)


def test_circular_arc_point_at_distance():
    center = [0, 0]
    v1 = [0, -1]
    angle = np.pi
    segment = fp.CircularArcSegment(center=center, point=v1, angle=angle)

    d = np.pi / 2
    p = segment.point_at_distance(d)
    assert np.allclose(p, [1, 0])
