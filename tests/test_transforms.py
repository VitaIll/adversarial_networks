"""Tests for the declarative parameter transforms (bijectors)."""

from __future__ import annotations

import math

import pytest
import torch

from adversarial_networks.transforms import Interval, Positive, Real


def test_real_is_identity() -> None:
    t = Real()
    raw = torch.tensor(2.3)
    assert torch.allclose(t.forward(raw), raw)
    assert abs(float(t.inverse(2.3)) - 2.3) < 1e-6
    assert t.default_constrained() == 0.0


def test_positive_round_trip_and_range() -> None:
    t = Positive()
    for v in (0.01, 1.0, 7.5, 100.0):
        assert abs(float(t.forward(t.inverse(v))) - v) < 1e-5
    assert float(t.forward(torch.tensor(-5.0))) > 0.0  # always positive
    assert t.default_constrained() == 1.0
    with pytest.raises(ValueError):
        t.inverse(-1.0)


def test_interval_round_trip_in_range_and_boundary_safe() -> None:
    t = Interval(-0.85, 0.85)
    for v in (-0.8, -0.4, 0.0, 0.4, 0.8):
        assert abs(float(t.forward(t.inverse(v))) - v) < 1e-4
    # strictly inside for any moderate raw (the regime an optimiser actually visits)
    moderate = t.forward(torch.tensor([-5.0, 5.0]))
    assert bool((moderate > -0.85).all() and (moderate < 0.85).all())
    # bounded within the closed interval even for extreme raw (float tanh saturates)
    extreme = t.forward(torch.tensor([-50.0, 50.0]))
    assert bool((extreme >= -0.85).all() and (extreme <= 0.85).all())
    # initialising at (or beyond) the boundary must not overflow to +/- inf
    assert math.isfinite(float(t.inverse(0.85)))
    assert math.isfinite(float(t.inverse(-0.85)))
    assert t.default_constrained() == 0.0


def test_interval_asymmetric() -> None:
    t = Interval(0.0, 4.0)
    assert abs(float(t.forward(torch.tensor(0.0))) - 2.0) < 1e-6  # center
    for v in (0.5, 2.0, 3.5):
        assert abs(float(t.forward(t.inverse(v))) - v) < 1e-4
    assert t.default_constrained() == 2.0


def test_interval_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError):
        Interval(1.0, 1.0)
    with pytest.raises(ValueError):
        Interval(1.0, 0.0)
