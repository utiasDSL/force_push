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

    info = path.compute_closest_point_info(p)
    assert np.isclose(info.deviation, 1.0)
    assert np.allclose(info.direction, [0, 1])
    assert np.allclose(info.point, [1, 1])
    assert np.isclose(info.distance_from_start, 2.0)
    assert np.isclose(info.offset, -1)

    # actual closest point is farther from the start than this, so there should
    # be no change
    info = path.compute_closest_point_info(p, min_dist_from_start=1.0)
    assert np.isclose(info.deviation, 1.0)
    assert np.allclose(info.direction, [0, 1])
    assert np.allclose(info.point, [1, 1])
    assert np.isclose(info.distance_from_start, 2.0)
    assert np.isclose(info.offset, -1)

    info = path.compute_closest_point_info(p, min_dist_from_start=3.0)
    assert np.isclose(info.deviation, np.sqrt(2))
    assert np.allclose(info.direction, [0, 1])
    assert np.allclose(info.point, [1, 2])
    assert np.isclose(info.distance_from_start, 3.0)
    assert np.isclose(info.offset, -1)


def test_line_path():
    path = fp.SegmentPath.line([1, 0])

    p = np.array([1, 1])

    info = path.compute_closest_point_info(p)
    assert np.isclose(info.deviation, 1.0)
    assert np.allclose(info.direction, [1, 0])
    assert np.allclose(info.point, [1, 0])
    assert np.isclose(info.distance_from_start, 1.0)
    assert np.isclose(info.offset, 1.0)


def test_segment_path_lines_and_arcs():
    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [3.0, 0]),
            fp.CircularArcSegment(center=[3.0, 2.0], point=[3.0, 0], angle=np.pi / 2),
            fp.LineSegment([5.0, 2], [5.0, 5], infinite=True),
        ],
    )
    cum_seg_lengths = np.cumsum([seg.length for seg in path.segments])

    # partway along the path (closest to the arc)
    p = np.array([5, 0])

    info = path.compute_closest_point_info(p)
    assert np.isclose(info.deviation, 2 * (np.sqrt(2) - 1))
    assert np.allclose(info.direction, [np.sqrt(2) / 2, np.sqrt(2) / 2])
    C = rotz(-np.pi / 4)[:2, :2]
    assert np.allclose(info.point, np.array([3, 2]) + C @ [2, 0])
    assert np.isclose(info.distance_from_start, cum_seg_lengths[0] + np.pi / 2)
    assert np.isclose(info.offset, -2 * (np.sqrt(2) - 1))

    # should be the same since min_dist_from_start is smaller than distance to
    # actual closest point
    info = path.compute_closest_point_info(p, min_dist_from_start=cum_seg_lengths[0])
    assert np.isclose(info.deviation, 2 * (np.sqrt(2) - 1))
    assert np.allclose(info.direction, [np.sqrt(2) / 2, np.sqrt(2) / 2])
    C = rotz(-np.pi / 4)[:2, :2]
    assert np.allclose(info.point, np.array([3, 2]) + C @ [2, 0])
    assert np.isclose(info.distance_from_start, cum_seg_lengths[0] + np.pi / 2)
    assert np.isclose(info.offset, -2 * (np.sqrt(2) - 1))

    # force the point to be farther along the path by specifying
    # min_dist_from_start
    info = path.compute_closest_point_info(p, min_dist_from_start=cum_seg_lengths[1])
    assert np.isclose(info.deviation, 2.0)
    assert np.allclose(info.direction, [0, 1])
    assert np.allclose(info.point, [5, 2])
    assert np.isclose(info.distance_from_start, cum_seg_lengths[1])
    assert np.isclose(info.offset, 0)

    # farther along the path (closest to the second line segment)
    p = np.array([4, 4])

    info = path.compute_closest_point_info(p)
    assert np.isclose(info.deviation, 1.0)
    assert np.allclose(info.direction, [0, 1])
    assert np.allclose(info.point, [5, 4])
    assert np.isclose(info.distance_from_start, cum_seg_lengths[1] + 2)
    assert np.isclose(info.offset, 1.0)


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

    path = fp.SegmentPath(
        [
            fp.LineSegment([0.0, 0], [3.0, 0]),
            fp.CircularArcSegment(center=[3.0, 2.0], point=[3.0, 0], angle=np.pi / 2),
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
    arm = np.array([0, -1])
    v1 = center + arm
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
    # this now depends on the minimum distance parameter
    info = segment.closest_point_info(center)
    assert np.allclose(info.point, segment.v1)
    assert np.isclose(info.deviation, segment.radius)

    θ = np.pi / 2
    min_dist = segment.radius * θ
    info = segment.closest_point_info(center, min_dist_from_start=min_dist)
    assert np.allclose(info.point, segment.center + fp.rot2d(θ) @ arm)
    assert np.isclose(info.deviation, segment.radius)

    # not allowed to specify a minimum distance greater than the length of the
    # segment
    min_dist = segment.radius * 1.1 * np.pi
    with pytest.raises(AssertionError):
        info = segment.closest_point_info(center, min_dist_from_start=min_dist)


def test_circular_arc_point_at_distance():
    center = [0, 0]
    v1 = [0, -1]
    angle = np.pi
    segment = fp.CircularArcSegment(center=center, point=v1, angle=angle)

    d = np.pi / 2
    p = segment.point_at_distance(d)
    assert np.allclose(p, [1, 0])
