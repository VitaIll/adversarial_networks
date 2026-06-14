"""Tests for the synthetic dataset factories (make_linear_in_means / make_effort_game)."""

from __future__ import annotations

import torch

from adversarial_networks.data import NetworkData
from adversarial_networks.datasets import make_effort_game, make_linear_in_means


def test_make_linear_in_means_returns_networkdata() -> None:
    data = make_linear_in_means(n_nodes=200, graph="ba", k=2, seed=0, m=2)
    assert isinstance(data, NetworkData)
    assert data.k == 2
    assert data.num_nodes <= 200  # sanitisation may drop a few nodes
    assert data.y.dtype == torch.float32 and bool(torch.isfinite(data.y).all())


def test_same_seed_gives_identical_outcome() -> None:
    a = make_linear_in_means(n_nodes=200, seed=3, m=2)
    b = make_linear_in_means(n_nodes=200, seed=3, m=2)
    assert torch.equal(a.y, b.y)
    assert torch.equal(a.X, b.X)


def test_different_seed_changes_outcome() -> None:
    a = make_linear_in_means(n_nodes=200, seed=3, m=2)
    c = make_linear_in_means(n_nodes=200, seed=4, m=2)
    assert not torch.equal(a.y, c.y)


def test_true_params_are_honoured() -> None:
    data = make_linear_in_means(n_nodes=300, seed=0, m=2, true_params={"beta": 0.3, "gamma": 2.0})
    # the simulated outcome should correlate strongly with X at gamma=2.0
    corr = torch.corrcoef(torch.stack([data.X, data.y]))[0, 1]
    assert float(corr) > 0.3


def test_make_effort_game_returns_networkdata_and_reproducible() -> None:
    a = make_effort_game(n_nodes=200, seed=1, m=2)
    b = make_effort_game(n_nodes=200, seed=1, m=2)
    assert isinstance(a, NetworkData)
    assert torch.equal(a.y, b.y)
    assert bool(torch.isfinite(a.y).all())
