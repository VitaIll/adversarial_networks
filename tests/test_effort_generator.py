"""Unit tests for the nonlinear effort-game structural generator."""

from __future__ import annotations

import functools
import math

import networkx as nx
import numpy as np
import pytest
import torch
from torch import Tensor
from torch_geometric.utils import from_networkx, to_undirected

from adversarial_networks.core.graph import row_stochastic_weights as build_row_stochastic_W
from adversarial_networks.generators import EffortGameGenerator


def _forward_fn_fixed_r(
    gamma: Tensor,
    raw_lambda: Tensor,
    log_mu: Tensor,
    log_sigma_sq: Tensor,
    eps: Tensor,
    *,
    W: Tensor,
    X: Tensor,
    lambda_max: float,
    fix_r: float,
    picard_max: int,
    newton_max: int,
) -> Tensor:
    """Functional fixed-iteration forward pass for gradcheck with fixed r."""
    lam = lambda_max * torch.sigmoid(raw_lambda)
    mu = torch.exp(log_mu)
    r = fix_r
    sigma = torch.sqrt(torch.exp(log_sigma_sq))

    Y = torch.zeros_like(X)
    for _ in range(picard_max):
        WY = torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)
        b = lam * WY + gamma * X + sigma * eps
        z = b / (1.0 + lam)
        for _ in range(newton_max):
            exp_neg_rz = torch.exp(-r * z)
            f_val = (1.0 + lam) * z - mu * r * exp_neg_rz - b
            f_prime = (1.0 + lam) + mu * r * r * exp_neg_rz
            z = z - (f_val / f_prime)
        Y = z
    return Y


def _forward_fn_free_r(
    gamma: Tensor,
    raw_lambda: Tensor,
    log_mu: Tensor,
    log_r: Tensor,
    log_sigma_sq: Tensor,
    eps: Tensor,
    *,
    W: Tensor,
    X: Tensor,
    lambda_max: float,
    picard_max: int,
    newton_max: int,
) -> Tensor:
    """Functional fixed-iteration forward pass for gradcheck with learnable r."""
    lam = lambda_max * torch.sigmoid(raw_lambda)
    mu = torch.exp(log_mu)
    r = torch.exp(log_r)
    sigma = torch.sqrt(torch.exp(log_sigma_sq))

    Y = torch.zeros_like(X)
    for _ in range(picard_max):
        WY = torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)
        b = lam * WY + gamma * X + sigma * eps
        z = b / (1.0 + lam)
        for _ in range(newton_max):
            exp_neg_rz = torch.exp(-r * z)
            f_val = (1.0 + lam) * z - mu * r * exp_neg_rz - b
            f_prime = (1.0 + lam) + mu * r * r * exp_neg_rz
            z = z - (f_val / f_prime)
        Y = z
    return Y


@pytest.fixture
def small_graph() -> tuple[Tensor, int]:
    """Connected graph with 20 nodes and sparse COO float32 W."""
    graph = nx.barabasi_albert_graph(20, 2, seed=0)
    n_nodes = graph.number_of_nodes()
    edge_index = from_networkx(graph).edge_index
    edge_index = to_undirected(edge_index, num_nodes=n_nodes).contiguous()
    W = build_row_stochastic_W(edge_index=edge_index, num_nodes=n_nodes)
    return W, n_nodes


@pytest.fixture
def gen() -> EffortGameGenerator:
    """Default fixed-r effort-game generator."""
    return EffortGameGenerator(fix_r=1.0, init_lambda=0.5, init_mu=0.1)


class TestConstructorValidation:
    def test_valid_construction_fixed_r(self) -> None:
        gen = EffortGameGenerator(fix_r=1.0, init_lambda=0.5, init_mu=0.1)
        assert len(list(gen.parameters())) == 3

    def test_valid_construction_free_r(self) -> None:
        gen = EffortGameGenerator(fix_r=None, init_r=1.0, init_lambda=0.5, init_mu=0.1)
        assert len(list(gen.parameters())) == 4

    def test_valid_construction_trainable_sigma(self) -> None:
        gen = EffortGameGenerator(
            fix_r=1.0,
            fix_sigma_sq=None,
            init_lambda=0.5,
            init_mu=0.1,
        )
        assert len(list(gen.parameters())) == 4

    def test_rejects_lambda_exceeding_cap(self) -> None:
        with pytest.raises(ValueError, match="init_lambda"):
            EffortGameGenerator(init_lambda=5.0, lambda_max=4.0)

    def test_rejects_nonpositive_mu(self) -> None:
        with pytest.raises(ValueError, match="init_mu"):
            EffortGameGenerator(init_mu=-1.0)

    def test_rejects_init_log_sigma_when_sigma_fixed(self) -> None:
        with pytest.raises(ValueError, match="init_log_sigma_sq"):
            EffortGameGenerator(fix_sigma_sq=1.0, init_log_sigma_sq=-0.2)

    def test_fixed_iterations_flag(self) -> None:
        gen = EffortGameGenerator(fixed_iterations=True, init_lambda=0.5, init_mu=0.1)
        assert gen.fixed_iterations is True


class TestForwardContract:
    def test_output_shape_dtype_device(self, gen: EffortGameGenerator, small_graph: tuple[Tensor, int]) -> None:
        W, n_nodes = small_graph
        Y = gen(W, torch.randn(n_nodes, dtype=torch.float32))
        assert Y.shape == (n_nodes,)
        assert Y.dtype == torch.float32
        assert Y.device == W.device

    def test_iteration_counts_recorded(
        self, gen: EffortGameGenerator, small_graph: tuple[Tensor, int]
    ) -> None:
        W, n_nodes = small_graph
        _ = gen(W, torch.randn(n_nodes, dtype=torch.float32))
        assert 1 <= gen.last_picard_iterations <= gen.picard_max
        assert 1 <= gen.last_newton_max_iters <= gen.newton_max

    def test_fixed_iterations_runs_all_steps(self, small_graph: tuple[Tensor, int]) -> None:
        W, n_nodes = small_graph
        gen = EffortGameGenerator(
            fixed_iterations=True,
            picard_max=10,
            newton_max=4,
            init_lambda=0.5,
            init_mu=0.1,
        )
        _ = gen(W, torch.randn(n_nodes, dtype=torch.float32))
        assert gen.last_picard_iterations == 10
        assert gen.last_newton_max_iters == 4


class TestEquilibriumCorrectness:
    def test_foc_residual(self, small_graph: tuple[Tensor, int]) -> None:
        """Returned equilibrium approximately satisfies nodewise FOCs."""
        W, n_nodes = small_graph
        X = torch.randn(n_nodes, dtype=torch.float32)
        gen = EffortGameGenerator(
            fix_r=1.0,
            fix_sigma_sq=None,
            picard_tol=1e-8,
            picard_max=200,
            newton_tol=1e-12,
            newton_max=20,
            init_gamma=1.2,
            init_lambda=0.7,
            init_mu=0.3,
            init_log_sigma_sq=-0.2,
        )

        torch.manual_seed(123)
        Y = gen(W, X)
        params = gen.get_params()

        torch.manual_seed(123)
        sigma = math.sqrt(params["sigma_sq"])
        eps = sigma * torch.randn_like(X)

        WY = torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)
        residual = (
            (1.0 + params["lambda_"]) * Y
            - params["lambda_"] * WY
            - params["mu"] * params["r"] * torch.exp(-params["r"] * Y)
            - (params["gamma"] * X + eps)
        )
        assert residual.abs().max().item() < 2e-4


class TestGradientFlowDiagnostics:
    def test_gradients_reach_all_parameters(
        self, gen: EffortGameGenerator, small_graph: tuple[Tensor, int]
    ) -> None:
        """loss.backward() populates finite gradients on every parameter."""
        W, n_nodes = small_graph
        Y = gen(W, torch.randn(n_nodes, dtype=torch.float32))
        Y.sum().backward()

        for name, param in gen.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"
        assert any(param.grad.abs().max().item() > 0.0 for param in gen.parameters())

    def test_fixed_sigma_excluded_from_parameters(self, gen: EffortGameGenerator) -> None:
        """Default fixed sigma_sq should not appear as a trainable parameter."""
        param_names = {name for name, _ in gen.named_parameters()}
        assert "log_sigma_sq" not in param_names

    def test_trainable_sigma_receives_gradient(
        self, small_graph: tuple[Tensor, int]
    ) -> None:
        """When unlocked, log_sigma_sq should receive finite gradient."""
        W, n_nodes = small_graph
        gen = EffortGameGenerator(fix_sigma_sq=None, init_lambda=0.5, init_mu=0.1)
        gen(W, torch.randn(n_nodes, dtype=torch.float32)).sum().backward()
        assert gen.log_sigma_sq.grad is not None
        assert torch.isfinite(gen.log_sigma_sq.grad).all()

    def test_no_gradient_explosion(
        self, gen: EffortGameGenerator, small_graph: tuple[Tensor, int]
    ) -> None:
        W, n_nodes = small_graph
        gen(W, torch.randn(n_nodes, dtype=torch.float32)).sum().backward()
        for name, param in gen.named_parameters():
            assert param.grad.abs().max().item() < 1e6, f"Explosion in {name}"

    def test_clamp_guard_prevents_overflow(self, small_graph: tuple[Tensor, int]) -> None:
        """Extreme initialization activates clamp guard but keeps outputs finite."""
        torch.manual_seed(0)
        W, n_nodes = small_graph
        gen = EffortGameGenerator(
            fix_r=1.0,
            fix_sigma_sq=None,
            init_gamma=1e4,
            init_lambda=0.5,
            init_mu=0.1,
            init_log_sigma_sq=-50.0,
        )
        X = torch.full((n_nodes,), -1e4, dtype=torch.float32)
        lam = gen.get_params()["lambda_"]
        r_val = gen.get_params()["r"]
        z0 = (gen.get_params()["gamma"] * X) / (1.0 + lam)
        assert z0.abs().max().item() > 50.0 / r_val
        Y = gen(W, X)
        assert torch.isfinite(Y).all()

    def test_gradient_changes_with_parameter(self, small_graph: tuple[Tensor, int]) -> None:
        """Changing lambda changes the simulated equilibrium under same noise draw."""
        W, n_nodes = small_graph
        X = torch.randn(n_nodes, dtype=torch.float32)
        gen_lo = EffortGameGenerator(fix_r=1.0, init_lambda=0.3, init_mu=0.1)
        gen_hi = EffortGameGenerator(fix_r=1.0, init_lambda=2.0, init_mu=0.1)

        torch.manual_seed(77)
        Y_lo = gen_lo(W, X)
        torch.manual_seed(77)
        Y_hi = gen_hi(W, X)

        max_abs_diff = (Y_lo - Y_hi).abs().max().item()
        assert max_abs_diff > 1e-4


class TestGetParams:
    def test_keys_and_types(self, gen: EffortGameGenerator) -> None:
        params = gen.get_params()
        assert set(params.keys()) == {"gamma", "lambda_", "mu", "r", "sigma_sq"}
        assert all(isinstance(value, float) for value in params.values())

    def test_constraints(self, gen: EffortGameGenerator) -> None:
        params = gen.get_params()
        assert 0.0 < params["lambda_"] < gen.lambda_max
        assert params["mu"] > 0.0
        assert params["sigma_sq"] > 0.0


class TestContractionRate:
    def test_valid_range(self, gen: EffortGameGenerator) -> None:
        assert 0.0 < gen.contraction_rate < 1.0

    def test_matches_lambda(self, gen: EffortGameGenerator) -> None:
        lam = gen.get_params()["lambda_"]
        assert abs(gen.contraction_rate - lam / (1.0 + lam)) < 1e-6


class TestInputValidation:
    def test_rejects_non_sparse_W(self, gen: EffortGameGenerator) -> None:
        with pytest.raises(TypeError, match="sparse"):
            gen(torch.eye(5, dtype=torch.float32), torch.randn(5, dtype=torch.float32))

    def test_rejects_shape_mismatch(
        self, gen: EffortGameGenerator, small_graph: tuple[Tensor, int]
    ) -> None:
        W, n_nodes = small_graph
        with pytest.raises(ValueError, match="mismatch"):
            gen(W, torch.randn(n_nodes + 1, dtype=torch.float32))


class TestGradientCorrectness:
    """Verify autograd gradients match finite differences through Picard+Newton."""

    @staticmethod
    def _make_small_graph_f64(n: int = 8) -> Tensor:
        """Create a tiny connected sparse row-stochastic matrix in float64."""
        graph = nx.cycle_graph(n)
        rng = np.random.default_rng(0)
        target_extra = max(2, n // 3)
        added = 0
        while added < target_extra:
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v or graph.has_edge(u, v):
                continue
            graph.add_edge(u, v)
            added += 1

        edge_index = from_networkx(graph).edge_index
        edge_index = to_undirected(edge_index, num_nodes=n).contiguous()
        W32 = build_row_stochastic_W(edge_index=edge_index, num_nodes=n).coalesce()
        return torch.sparse_coo_tensor(
            W32.indices(),
            W32.values().to(torch.float64),
            W32.shape,
            dtype=torch.float64,
        ).coalesce()

    def test_gradcheck_fixed_r(self) -> None:
        """Autograd gradient matches finite differences (fixed r)."""
        n_nodes = 8
        W = self._make_small_graph_f64(n_nodes)
        X = torch.randn(n_nodes, dtype=torch.float64)
        eps = torch.randn(n_nodes, dtype=torch.float64)

        gamma = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        raw_lambda = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        log_mu = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
        log_sigma_sq = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        eps_input = eps.clone().requires_grad_(True)

        fn = functools.partial(
            _forward_fn_fixed_r,
            W=W,
            X=X,
            lambda_max=4.0,
            fix_r=1.0,
            picard_max=15,
            newton_max=5,
        )
        assert torch.autograd.gradcheck(
            fn,
            (gamma, raw_lambda, log_mu, log_sigma_sq, eps_input),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradcheck_free_r(self) -> None:
        """Autograd gradient matches finite differences (learnable r)."""
        n_nodes = 8
        W = self._make_small_graph_f64(n_nodes)
        X = torch.randn(n_nodes, dtype=torch.float64)
        eps = torch.randn(n_nodes, dtype=torch.float64)

        gamma = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        raw_lambda = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        log_mu = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
        log_r = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        log_sigma_sq = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        eps_input = eps.clone().requires_grad_(True)

        fn = functools.partial(
            _forward_fn_free_r,
            W=W,
            X=X,
            lambda_max=4.0,
            picard_max=15,
            newton_max=5,
        )
        assert torch.autograd.gradcheck(
            fn,
            (gamma, raw_lambda, log_mu, log_r, log_sigma_sq, eps_input),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
