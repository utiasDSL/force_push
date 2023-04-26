import pytest
import numpy as np

from mmpush.util import *


def test_signed_angle_ccw():
    a = np.array([1, 0])
    b = np.array([0, 1])
    θ = signed_angle(a, b)
    assert np.isclose(θ, np.pi / 2)


def test_signed_angle_cw():
    a = np.array([1, 0])
    b = np.array([0, -1])
    θ = signed_angle(a, b)
    assert np.isclose(θ, -np.pi / 2)


def test_signed_angle_cw():
    a = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    b = np.array([np.cos(3 * np.pi / 4), np.sin(3 * np.pi / 4)])
    θ = signed_angle(a, b)
    assert np.isclose(θ, np.pi / 2)


def test_wrap_to_pi():
    # postive wraps to negative
    x = np.pi + 1
    assert np.isclose(wrap_to_pi(x), x - 2 * np.pi)

    # negative wraps to positive
    x = -np.pi - 1
    assert np.isclose(wrap_to_pi(x), x + 2 * np.pi)

    # test random numbers in range [-50, 50)
    np.random.seed(0)
    for _ in range(5):
        x = 100 * np.random.random() - 50
        assert -np.pi <= wrap_to_pi(x) <= np.pi


def test_perp2d():
    np.random.seed(0)
    C = rot2d(np.pi / 2)

    # test various random vectors
    for _ in range(5):
        x = 2 * np.random.random(2) - 1
        assert np.allclose(C @ x, perp2d(x))
