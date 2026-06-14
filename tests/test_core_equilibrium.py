"""Tests for the differentiable equilibrium core (picard / newton / solve_equilibrium).

Pins the new numeric kernels in isolation (float64): Picard converges to the dense
linear solve, the analytic and AD-diagonal Newton agree, and the ``"unroll"`` and
``"implicit"`` structural-gradient strategies agree (value and gradient) at a
converged fixed point, with ``gradcheck`` on the unrolled solve.
"""

from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.utils import from_networkx, to_undirected

from adversarial_networks.core.equilibrium import newton, picard, solve_equilibrium
from adversarial_networks.core.graph import row_stochastic_weights


def _w64(n: int = 12, seed: int = 0) -> torch.Tensor:
    graph = nx.barabasi_albert_graph(n, 2, seed=seed)
    ei = to_undirected(from_networkx(graph).edge_index, num_nodes=n).contiguous()
    return row_stochastic_weights(ei, n).to(torch.float64)


def test_picard_matches_dense_linear_solve() -> None:
    torch.manual_seed(0)
    n = 12
    W = _w64(n)
    X = torch.randn(n, dtype=torch.float64)
    shocks = torch.randn(n, dtype=torch.float64)
    beta, gamma = torch.tensor(0.5, dtype=torch.float64), torch.tensor(1.3, dtype=torch.float64)
    base = gamma * X + shocks

    def step(Y):
        return beta * torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1) + base

    Y, iters = picard(step, torch.zeros(n, dtype=torch.float64), tol=1e-13, max_iter=500)
    dense = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - beta * W.to_dense(), base)
    assert (Y - dense).abs().max().item() < 1e-9
    assert 1 <= iters <= 500


def test_newton_analytic_equals_ad_diagonal() -> None:
    n = 10
    torch.manual_seed(1)
    X = torch.randn(n, dtype=torch.float64)
    b = 1.3 * X + torch.randn(n, dtype=torch.float64)
    lam, mu, r = (torch.tensor(v, dtype=torch.float64) for v in (0.8, 0.4, 1.0))

    def residual(z):
        return (1 + lam) * z - mu * r * torch.exp(-r * z) - b

    def jacobian(z):
        return (1 + lam) + mu * r * r * torch.exp(-r * z)

    z0 = b / (1 + lam)
    z_analytic, _ = newton(residual, z0, tol=1e-13, max_iter=40, jacobian_fn=jacobian)
    z_ad, _ = newton(residual, z0, tol=1e-13, max_iter=40, jacobian_fn=None)
    assert (z_analytic - z_ad).abs().max().item() < 1e-10
    assert residual(z_analytic).abs().max().item() < 1e-10


def test_unroll_and_implicit_agree_on_value_and_gradient() -> None:
    n = 10
    W = _w64(n, seed=2)
    torch.manual_seed(2)
    X = torch.randn(n, dtype=torch.float64)
    shocks = torch.randn(n, dtype=torch.float64)
    weights = torch.linspace(0.5, 1.5, n, dtype=torch.float64)

    lam = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def apply_step(Y):
        b = lam * torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1) + gamma * X + shocks
        z = b / (1 + lam)
        for _ in range(14):
            e = torch.exp(-z)
            z = z - ((1 + lam) * z - mu * e - b) / ((1 + lam) + mu * e)
        return z

    def run(mode):
        for p in (lam, gamma, mu):
            p.grad = None
        Y, _ = solve_equilibrium(apply_step, torch.zeros(n, dtype=torch.float64),
                                 tol=1e-13, max_iter=300, differentiation=mode)
        (Y * weights).sum().backward()
        return Y.detach(), [float(p.grad) for p in (lam, gamma, mu)]

    y_unroll, g_unroll = run("unroll")
    y_implicit, g_implicit = run("implicit")
    assert (y_unroll - y_implicit).abs().max().item() < 1e-9
    assert max(abs(a - b) for a, b in zip(g_unroll, g_implicit)) < 1e-7


def test_gradcheck_unrolled_solve() -> None:
    n = 8
    W = _w64(n, seed=3)
    torch.manual_seed(3)
    X = torch.randn(n, dtype=torch.float64)
    shocks = torch.randn(n, dtype=torch.float64)

    def f(beta, gamma):
        base = gamma * X + shocks

        def step(Y):
            return beta * torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1) + base

        Y, _ = solve_equilibrium(step, torch.zeros(n, dtype=torch.float64),
                                 tol=0.0, max_iter=80, differentiation="unroll", fixed_iterations=True)
        return Y

    beta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (beta, gamma), eps=1e-6, atol=1e-5, rtol=1e-4)
