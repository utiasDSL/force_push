import pytest
import numpy as np

from mmpush import *


np.set_printoptions(suppress=True)


def compare_motions(f_max, τ_max, μ, r_co_o, nc, vp):
    """Solve pusher-slider motion using both analytical and QP-based
    formulations and ensure they are correct.

    Returns a tuple (V, f, α), where V is the object velocity (in the body
    frame), f is the contact force (in the body frame), and α is the slip
    velocity (perpendicular to the contact normal).
    """
    W = np.array([[1, 0], [0, 1], [-r_co_o[1], r_co_o[0]]])
    motion1 = PusherSliderMotion(f_max, τ_max, μ)
    motion2 = QPPusherSliderMotion(f_max, τ_max, μ)

    V1, f1, α1 = motion1.solve(vp, r_co_o, nc)
    V2, f2, α2 = motion2.solve(vp, r_co_o, nc)

    assert np.allclose(V1, V2)
    assert np.allclose(f1, f2)
    assert np.isclose(α1, α2)

    M = limit_surface_ellipsoid(f_max, τ_max)
    τ = perp2d(r_co_o) @ f1
    F = np.append(f1, τ)

    assert np.isclose(F @ M @ F, 1)

    return V1, f1, α1


def test_push_straight():
    """Compare motions with a straight-on push through the CoF."""
    f_max = 5
    τ_max = 1
    μ = 0.5

    r_co_o = np.array([-0.5, -0])
    nc = np.array([1, 0])

    vp = np.array([1, 0])
    compare_motions(f_max, τ_max, μ, r_co_o, nc, vp)


def test_push_angle():
    """Compare motions with a push at an angle."""
    f_max = 5
    τ_max = 1
    μ = 0.5

    r_co_o = np.array([-0.5, -0])
    nc = np.array([1, 0])

    vp = rot2d(np.pi / 4) @ np.array([1, 0])
    compare_motions(f_max, τ_max, μ, r_co_o, nc, vp)


def test_slip():
    """Compare motions with a push that produces slip."""
    f_max = 5
    τ_max = 1
    μ = 0.1

    r_co_o = np.array([-0.5, -0])
    nc = np.array([1, 0])

    vp = rot2d(np.pi / 4) @ np.array([1, 0])
    _, _, α = compare_motions(f_max, τ_max, μ, r_co_o, nc, vp)

    # ensure there is some nontrivial slip
    assert np.abs(α) > 0.1


def test_loss_of_contact():
    """Test when the pusher is actually pulling away from the slider."""
    f_max = 5
    τ_max = 1
    μ = 0.1

    r_co_o = np.array([-0.5, -0])
    nc = np.array([1, 0])
    W = np.array([[1, 0], [0, 1], [-r_co_o[1], r_co_o[0]]])
    vp = np.array([-1, 0])

    motion1 = PusherSliderMotion(f_max, τ_max, μ)
    motion2 = QPPusherSliderMotion(f_max, τ_max, μ)

    with pytest.raises(ValueError):
        V1, f1, α1 = motion1.solve(vp, r_co_o, nc)

    with pytest.raises(ValueError):
        V2, f2, α2 = motion2.solve(vp, r_co_o, nc)
