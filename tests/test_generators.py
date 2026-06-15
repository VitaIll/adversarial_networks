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

from adversarial_networks.core.equilibrium import EquilibriumNotConverged
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


# ------------------------------------------------- one-time hook post-condition guard
def test_forward_rejects_wrong_shape_best_response() -> None:
    """A ``best_response`` that returns a wrong-shape tensor (here ``(n, 2)``) must trip
    the one-time boundary post-condition inside ``forward`` — an attributable ValueError
    naming the hook and the model class, not a cryptic failure many Picard steps later."""
    class BadShapeGame(NetworkGameGenerator):
        beta = Interval(-0.8, 0.8)
        sigma_sq = Positive()

        def best_response(self, peer_agg, X, shocks):
            out = self.params()["beta"] * peer_agg + shocks
            return out.unsqueeze(-1).expand(-1, 2)  # wrong shape (n, 2)

    n = 20
    W = _w(n, seed=6)
    X = torch.randn(n)
    model = BadShapeGame()
    with pytest.raises(ValueError, match="best_response"):
        model(W, X)
    # the message also attributes the failing model class
    try:
        model(W, X)
    except ValueError as exc:
        assert "BadShapeGame" in str(exc)


def test_forward_rejects_grad_disconnected_best_response() -> None:
    """When a parameter requires grad, a ``best_response`` that detaches its output
    breaks the structural gradient; the post-condition must reject it (and stay silent
    under ``no_grad``, where the check is correctly gated)."""
    class DetachGame(NetworkGameGenerator):
        beta = Interval(-0.8, 0.8)
        sigma_sq = Positive()

        def best_response(self, peer_agg, X, shocks):
            return (self.params()["beta"] * peer_agg + shocks).detach()

    n = 15
    W = _w(n, seed=7)
    X = torch.randn(n)
    model = DetachGame()
    with pytest.raises(ValueError, match="grad-connected"):
        model(W, X)
    with torch.no_grad():  # gated off: no grad to connect, so no post-condition failure
        Y = model(W, X)
    assert Y.shape == (n,)


def test_forward_rejects_scalar_grad_connected_foc_residual() -> None:
    """A grad-connected SCALAR foc_residual broadcasts through Newton's z - g/g' back to
    the (n,) iterate, so the forward post-condition (which inspects Newton's OUTPUT) cannot
    catch it — the primitive's first-iteration shape guard on the RAW residual must (D2-03).
    """
    class ScalarFocGame(NetworkGameGenerator):
        gamma = Real()
        sigma_sq = Positive()

        def foc_residual(self, y, peer_agg, X, shocks):
            # Collapses (n,) -> scalar but stays grad-connected to gamma.
            return (y - self.params()["gamma"] * X - shocks).sum()

    n = 12
    W = _w(n, seed=11)
    X = torch.randn(n)
    model = ScalarFocGame(initial_values={"gamma": 1.0, "sigma_sq": 1.0})
    with pytest.raises(ValueError, match="residual_fn"):
        model(W, X)


def test_raise_on_nonconvergence_raises_equilibrium_not_converged() -> None:
    """With raise_on_nonconvergence=True, a Picard solve that hits its cap without the tol
    test firing raises EquilibriumNotConverged carrying the residual/tol/cap."""
    n = 40
    W = _w(n, seed=12)
    X = torch.randn(n)
    # picard_max=1 cannot reach the beta=0.5 equilibrium from the zero start.
    model = LinearInMeansGenerator(
        beta_cap=0.85, init_beta=0.5, init_gamma=1.0, picard_max=1
    )
    model.raise_on_nonconvergence = True
    with pytest.raises(EquilibriumNotConverged) as exc:
        model(W, X)
    assert exc.value.max_iter == 1
    assert exc.value.residual >= exc.value.tol


def test_forward_records_picard_residual_and_converged() -> None:
    """A converged forward records last_picard_residual (< tol) and last_picard_converged."""
    n = 20
    W = _w(n, seed=13)
    X = torch.randn(n)
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5, picard_tol=1e-7)
    torch.manual_seed(0)
    model(W, X)
    assert model.last_picard_converged is True
    assert model.last_picard_residual < 1e-7


def test_scalar_builtins_reject_vector_covariate_attributably() -> None:
    """The scalar built-ins (Linear/Effort) fed a (n, d_x>=2) X must raise an attributable
    ValueError at the boundary (naming the class + the scalar-only scope), not a raw
    broadcast RuntimeError deep inside best_response (D7-REG-01)."""
    n = 20
    W = _w(n, seed=14)
    X2 = torch.randn(n, 2)
    for model in (
        LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5),
        EffortGameGenerator(init_lambda=0.5, init_mu=0.1),
    ):
        with pytest.raises(ValueError, match="scalar covariate only"):
            model(W, X2)
        # the message attributes the model class
        try:
            model(W, X2)
        except ValueError as exc:
            assert type(model).__name__ in str(exc)
    # a (n,) X still works (scalar path unchanged)
    Y = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)(W, torch.randn(n))
    assert Y.shape == (n,)


def test_builtin_forward_passes_hook_postcondition() -> None:
    """The built-in generators satisfy the post-conditions: a forward runs fine and is
    grad-connected (the checks are designed to pass for valid models)."""
    n = 20
    W = _w(n, seed=8)
    X = torch.randn(n)
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    torch.manual_seed(0)
    Y = model(W, X)
    assert Y.shape == (n,) and bool(torch.isfinite(Y).all())
    assert Y.requires_grad  # grad-connected through the solve


def test_builtin_get_params_keys_and_contraction_rate() -> None:
    lin = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    assert set(lin.get_params()) == {"beta", "gamma", "sigma_sq"}
    eff = EffortGameGenerator(init_lambda=0.5, init_mu=0.1)
    assert set(eff.get_params()) == {"gamma", "lambda_", "mu", "r", "sigma_sq"}
    assert 0.0 < eff.contraction_rate < 1.0
    assert math.isclose(eff.contraction_rate, 0.5 / 1.5, rel_tol=1e-6)


# ---------------------------------------------- generator-level Picard integration
def test_picard_matches_dense_solve() -> None:
    """The generator's full forward (shock draw + ``beta_cap·tanh`` reparam + Picard)
    matches the dense ``(I - beta·W)^{-1}`` solve. This is an *integration* check of the
    shock-seeding contract that the bare-``picard`` core unit test does not exercise
    (relocated from the retired ``test_utils.py``)."""
    torch.manual_seed(123)
    n = 10
    beta = 0.3
    gamma = 1.0
    sigma_sq = 1.0

    graph = nx.path_graph(n)
    edge_index = to_undirected(from_networkx(graph).edge_index, num_nodes=n)
    W = row_stochastic_weights(edge_index=edge_index, num_nodes=n)
    X = torch.randn(n, dtype=torch.float32)

    generator = LinearInMeansGenerator(
        beta_cap=0.9,
        picard_tol=1e-10,
        picard_max=300,
        init_beta=beta,
        init_gamma=gamma,
        init_log_sigma_sq=math.log(sigma_sq),
    )

    torch.manual_seed(999)
    y_picard = generator(W, X)

    torch.manual_seed(999)
    z = torch.randn_like(X)
    eps = math.sqrt(sigma_sq) * z
    rhs = X * gamma + eps
    A = torch.eye(n, dtype=torch.float32) - beta * W.to_dense()
    y_dense = torch.linalg.solve(A, rhs)

    max_abs_error = torch.max(torch.abs(y_picard - y_dense)).item()
    assert max_abs_error <= 1e-5, (
        f"Generator Picard deviates from direct solve: max error {max_abs_error:.2e} "
        f"(picard iters {generator.last_picard_iterations})"
    )
