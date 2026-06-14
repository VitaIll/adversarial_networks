"""Tests for the NetworkGameGenerator framework base and the built-in games.

Covers the subclass contract (best_response xor foc_residual), declarative
``Transform`` auto-wiring, the default vs. custom peer aggregate, the FOC route
(AD-Newton), and the cross-validation that ``differentiation="implicit"`` agrees
with ``"unroll"`` on the effort game.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest
import torch
from torch_geometric.utils import from_networkx, to_undirected

from adversarial_networks.core.graph import row_stochastic_weights
from adversarial_networks.generators import (
    EffortGameGenerator,
    LinearInMeansGenerator,
    NetworkGameGenerator,
)
from adversarial_networks.transforms import Interval, Positive, Real


def _w(n: int = 30, seed: int = 0) -> torch.Tensor:
    graph = nx.barabasi_albert_graph(n, 2, seed=seed)
    ei = to_undirected(from_networkx(graph).edge_index, num_nodes=n).contiguous()
    return row_stochastic_weights(ei, n)


# ------------------------------------------------------------- subclass contract
def test_requires_exactly_one_of_best_response_or_foc() -> None:
    class Both(NetworkGameGenerator):
        sigma_sq = Positive()

        def best_response(self, peer_agg, X, shocks):
            return shocks

        def foc_residual(self, y, peer_agg, X, shocks):
            return y

    class Neither(NetworkGameGenerator):
        sigma_sq = Positive()

    with pytest.raises(TypeError):
        Both()
    with pytest.raises(TypeError):
        Neither()


def test_declarative_transforms_auto_wire_constrained_params() -> None:
    class Toy(NetworkGameGenerator):
        beta = Interval(-0.85, 0.85)
        gamma = Real()
        sigma_sq = Positive()

        def best_response(self, peer_agg, X, shocks):
            p = self.params()
            return p["beta"] * peer_agg + p["gamma"] * X + shocks

    g = Toy(initial_values={"beta": 0.4, "gamma": 1.5, "sigma_sq": 2.0})
    params = g.get_params()
    assert set(params) == {"beta", "gamma", "sigma_sq"}
    assert abs(params["beta"] - 0.4) < 1e-4
    assert abs(params["gamma"] - 1.5) < 1e-5
    assert abs(params["sigma_sq"] - 2.0) < 1e-4
    assert len(list(g.parameters())) == 3  # one learnable raw leaf per declared field

    g0 = Toy()  # default init: forward(0) per transform
    assert abs(g0.get_params()["beta"] - 0.0) < 1e-6
    assert abs(g0.get_params()["sigma_sq"] - 1.0) < 1e-6


def test_default_peer_aggregate_is_row_stochastic_mean() -> None:
    n = 20
    W = _w(n, seed=1)
    Y = torch.randn(n)
    g = LinearInMeansGenerator(beta_cap=0.85)
    expected = torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(g.peer_aggregate(W, Y), expected)


def test_custom_peer_aggregate_raw_sum() -> None:
    class RawSumGame(NetworkGameGenerator):
        beta = Interval(-0.3, 0.3)
        sigma_sq = Positive()

        def peer_aggregate(self, W, Y):
            degree = torch.bincount(W.coalesce().indices()[0], minlength=Y.shape[0]).to(Y.dtype)
            return torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1) * degree

        def best_response(self, peer_agg, X, shocks):
            p = self.params()
            return torch.tanh(p["beta"] * peer_agg) + shocks

    n = 25
    W = _w(n, seed=2)
    g = RawSumGame(picard_tol=1e-6, picard_max=200)
    torch.manual_seed(0)
    Y = g(W, torch.randn(n))
    assert Y.shape == (n,) and bool(torch.isfinite(Y).all())


# --------------------------------------------------------------------- FOC route
def test_foc_residual_route_matches_closed_form_linear() -> None:
    """A linear FOC ``y = beta*peer + gamma*X + eps`` solved by AD-Newton must match
    the closed-form linear-in-means equilibrium / dense solve."""
    class FocLinear(NetworkGameGenerator):
        beta = Interval(-0.85, 0.85)
        gamma = Real()
        sigma_sq = Positive()

        def foc_residual(self, y, peer_agg, X, shocks):
            p = self.params()
            return y - (p["beta"] * peer_agg + p["gamma"] * X + shocks)

    n = 20
    W = _w(n, seed=3)
    torch.manual_seed(5)
    X = torch.randn(n)
    g = FocLinear(initial_values={"beta": 0.4, "gamma": 1.3, "sigma_sq": 1.0},
                  picard_tol=1e-7, picard_max=300, newton_tol=1e-12, newton_max=10)
    torch.manual_seed(7)
    Y = g(W, X)
    torch.manual_seed(7)
    shocks = torch.randn(n)  # same draw as g's sample_shocks (sigma=1)
    base = 1.3 * X + shocks
    dense = torch.linalg.solve(torch.eye(n) - 0.4 * W.to_dense(), base)
    assert (Y - dense).abs().max().item() < 1e-4
    # gradients reach every learnable parameter
    g.zero_grad(set_to_none=True)
    g(W, X).sum().backward()
    for name, p in g.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name


# ---------------------------------------------------- implicit vs unroll on effort
def test_implicit_matches_unroll_on_effort_game() -> None:
    n = 24
    W = _w(n, seed=4)
    torch.manual_seed(9)
    X = torch.randn(n)
    weights = torch.linspace(0.4, 1.6, n)

    m_unroll = EffortGameGenerator(fix_r=1.0, fix_sigma_sq=1.0, init_gamma=1.2,
                                   init_lambda=0.7, init_mu=0.3, differentiation="unroll")
    m_impl = EffortGameGenerator(fix_r=1.0, fix_sigma_sq=1.0, init_gamma=1.2,
                                 init_lambda=0.7, init_mu=0.3, differentiation="implicit")
    m_impl.load_state_dict(m_unroll.state_dict())

    def run(model):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        Y = model(W, X)
        (Y * weights).sum().backward()
        return Y.detach(), {k: v.grad.clone() for k, v in model.named_parameters()}

    y_u, g_u = run(m_unroll)
    y_i, g_i = run(m_impl)
    assert (y_u - y_i).abs().max().item() < 1e-4
    for name in g_u:
        assert torch.allclose(g_u[name], g_i[name], rtol=1e-4, atol=1e-5), name


def test_builtin_get_params_keys_and_contraction_rate() -> None:
    lin = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    assert set(lin.get_params()) == {"beta", "gamma", "sigma_sq"}
    eff = EffortGameGenerator(init_lambda=0.5, init_mu=0.1)
    assert set(eff.get_params()) == {"gamma", "lambda_", "mu", "r", "sigma_sq"}
    assert 0.0 < eff.contraction_rate < 1.0
    assert math.isclose(eff.contraction_rate, 0.5 / 1.5, rel_tol=1e-6)
