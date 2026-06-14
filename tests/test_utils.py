"""Deterministic CPU tests for the MVP utility module.

All tests run on CPU with fixed seeds for reproducibility.
Enhanced with detailed assertions showing expected vs actual values.
"""

from __future__ import annotations

import math

import networkx as nx
import torch
from torch_geometric.utils import from_networkx, k_hop_subgraph, to_undirected

from adversarial_networks.config import InstanceNoiseConfig
from adversarial_networks.core.ego_features import extract_ego_batch
from adversarial_networks.core.graph import row_stochastic_weights as build_row_stochastic_W
from adversarial_networks.core.objective import instance_noise_taus as compute_instance_noise_taus
from adversarial_networks.generators import LinearInMeansGenerator as SCMGenerator


def _path_graph_edge_index(num_nodes: int) -> torch.Tensor:
    """Helper: Create edge_index for a path graph."""
    graph = nx.path_graph(num_nodes)
    data = from_networkx(graph)
    return to_undirected(data.edge_index, num_nodes=num_nodes)


def test_W_row_stochastic() -> None:
    """Test that W matrix is row-stochastic with proper shape and values."""
    n = 10
    edge_index = _path_graph_edge_index(n)
    W = build_row_stochastic_W(edge_index=edge_index, num_nodes=n)
    dense_W = W.to_dense()

    # Check shape
    assert W.shape == (n, n), f"Expected shape ({n}, {n}), got {W.shape}"

    # Check all values are finite
    assert torch.isfinite(dense_W).all(), "W contains non-finite values"

    # Check row-stochastic property
    row_sums = dense_W.sum(dim=1)
    max_deviation = torch.max(torch.abs(row_sums - 1.0)).item()
    tolerance = 1e-6

    assert torch.allclose(row_sums, torch.ones(n), atol=tolerance, rtol=0.0), (
        f"W is not row-stochastic:\n"
        f"  Expected: all row sums = 1.0\n"
        f"  Max deviation: {max_deviation:.2e}\n"
        f"  Tolerance: {tolerance:.2e}\n"
        f"  Row sums: {row_sums.tolist()}"
    )


def test_picard_matches_dense_solve() -> None:
    """Test that Picard iteration matches direct (I - βW)^(-1) solve."""
    torch.manual_seed(123)
    n = 10
    beta = 0.3
    gamma = 1.0
    sigma_sq = 1.0

    edge_index = _path_graph_edge_index(n)
    W = build_row_stochastic_W(edge_index=edge_index, num_nodes=n)
    X = torch.randn(n, dtype=torch.float32)

    generator = SCMGenerator(
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
    tolerance = 1e-5

    assert max_abs_error <= tolerance, (
        f"Picard solution deviates from direct solve:\n"
        f"  Expected max error: ≤ {tolerance:.2e}\n"
        f"  Actual max error:   {max_abs_error:.2e}\n"
        f"  Picard iterations:  {generator.last_picard_iterations}\n"
        f"  ||y_picard||:       {torch.norm(y_picard).item():.4f}\n"
        f"  ||y_dense||:        {torch.norm(y_dense).item():.4f}\n"
        f"  Relative error:     {max_abs_error / torch.norm(y_dense).item():.2e}"
    )


def test_generator_grad_flows_to_all_params() -> None:
    """Test that gradients flow to all three generator parameters."""
    torch.manual_seed(7)
    n = 12
    edge_index = _path_graph_edge_index(n)
    W = build_row_stochastic_W(edge_index=edge_index, num_nodes=n)
    X = torch.randn(n, dtype=torch.float32)

    generator = SCMGenerator(
        beta_cap=0.8,
        picard_tol=1e-8,
        picard_max=200,
        init_beta=0.1,
        init_gamma=0.2,
        init_log_sigma_sq=-0.2,
    )
    y_sim = generator(W, X)
    loss = y_sim.square().mean()
    loss.backward()

    for name in ("raw_beta", "gamma", "log_sigma_sq"):
        param = getattr(generator, name)
        grad = param.grad

        assert grad is not None, (
            f"Gradient is None for parameter '{name}'\n"
            f"  Expected: non-None gradient tensor\n"
            f"  Actual:   None\n"
            f"  This indicates gradient flow is broken!"
        )

        grad_magnitude = grad.abs().item()
        assert grad_magnitude > 0.0, (
            f"Gradient is zero for parameter '{name}'\n"
            f"  Expected: |grad| > 0\n"
            f"  Actual:   |grad| = {grad_magnitude:.2e}\n"
            f"  Parameter value: {param.item():.4f}\n"
            f"  Loss value: {loss.item():.4f}"
        )


def test_extract_ego_batch_root_marker() -> None:
    """Test that root markers are correctly set in batched ego-subgraphs."""
    torch.manual_seed(11)
    n = 14
    edge_index = _path_graph_edge_index(n)

    ego_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for root in range(n):
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=root,
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=n,
        )
        ego_cache[root] = (subset, sub_edge_index, int(mapping.item()))

    roots = torch.tensor([0, 3, 7, 10], dtype=torch.long)
    X = torch.randn(n, dtype=torch.float32)
    Y = torch.randn(n, dtype=torch.float32)
    norm_stats = {
        "mu_X": float(X.mean().item()),
        "sigma_X": float(X.std(unbiased=False).item()),
        "mu_Y": float(Y.mean().item()),
        "sigma_Y": float(Y.std(unbiased=False).item()),
    }

    batch, root_indices = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats,
    )

    # Check feature dimension
    expected_features = 3  # [X_tilde, Y_tilde, root_marker]
    assert batch.x.shape[1] == expected_features, (
        f"Expected {expected_features} node features, got {batch.x.shape[1]}"
    )

    root_marker = batch.x[:, 2]

    # Check each subgraph has exactly one root marked
    for idx in range(roots.numel()):
        start = int(batch.ptr[idx].item())
        end = int(batch.ptr[idx + 1].item())
        markers = root_marker[start:end]
        marker_sum = markers.sum().item()

        assert torch.isclose(markers.sum(), torch.tensor(1.0), atol=1e-6, rtol=0.0), (
            f"Subgraph {idx} (root={roots[idx].item()}) has incorrect root marker sum:\n"
            f"  Expected: exactly 1.0 (one root per subgraph)\n"
            f"  Actual:   {marker_sum:.6f}\n"
            f"  Subgraph size: {end - start} nodes"
        )

        # Check root_indices points to the marked node
        root_global = int(root_indices[idx].item())
        assert start <= root_global < end, (
            f"root_indices[{idx}] = {root_global} is out of range [{start}, {end})"
        )
        assert torch.isclose(
            root_marker[root_global], torch.tensor(1.0), atol=1e-6, rtol=0.0
        ), (
            f"Node at root_indices[{idx}] = {root_global} is not marked as root:\n"
            f"  Expected: marker = 1.0\n"
            f"  Actual:   marker = {root_marker[root_global].item():.6f}"
        )


def test_extract_ego_batch_blur_disabled_matches_baseline() -> None:
    """With blur disabled, feature construction is exactly unchanged."""
    torch.manual_seed(1234)
    n = 16
    edge_index = _path_graph_edge_index(n)

    ego_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for root in range(n):
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=root,
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=n,
        )
        ego_cache[root] = (subset, sub_edge_index, int(mapping.item()))

    roots = torch.tensor([1, 4, 9, 12], dtype=torch.long)
    X = torch.randn(n, dtype=torch.float32)
    Y = torch.randn(n, dtype=torch.float32)
    norm_stats = {
        "mu_X": float(X.mean().item()),
        "sigma_X": float(X.std(unbiased=False).item()),
        "mu_Y": float(Y.mean().item()),
        "sigma_Y": float(Y.std(unbiased=False).item()),
    }

    batch_base, idx_base = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats,
    )

    blur_cfg = InstanceNoiseConfig(enabled=False, tau_x0=0.07, tau_y0=0.12)
    batch_disabled, idx_disabled = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats,
        instance_noise=blur_cfg,
        generator_step=37,
        batch_role="real",
    )

    assert torch.equal(batch_base.x, batch_disabled.x)
    assert torch.equal(batch_base.edge_index, batch_disabled.edge_index)
    assert torch.equal(idx_base, idx_disabled)


def test_extract_ego_batch_blur_enabled_changes_xy_not_root_marker() -> None:
    """Enabled blur perturbs X/Y features while preserving root marker exactly."""
    torch.manual_seed(2025)
    n = 32
    edge_index = _path_graph_edge_index(n)

    ego_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for root in range(n):
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=root,
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=n,
        )
        ego_cache[root] = (subset, sub_edge_index, int(mapping.item()))

    roots = torch.arange(0, n, 2, dtype=torch.long)
    X = torch.randn(n, dtype=torch.float32)
    Y = torch.randn(n, dtype=torch.float32)
    norm_stats = {
        "mu_X": float(X.mean().item()),
        "sigma_X": float(X.std(unbiased=False).item()),
        "mu_Y": float(Y.mean().item()),
        "sigma_Y": float(Y.std(unbiased=False).item()),
    }

    batch_base, idx_base = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats,
        batch_role="fake",
    )

    blur_cfg = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.05,
        tau_y0=0.10,
        schedule="constant",
        anneal_steps=0,
    )
    torch.manual_seed(4040)
    batch_blur, idx_blur = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats,
        instance_noise=blur_cfg,
        generator_step=12,
        batch_role="fake",
    )
    torch.manual_seed(4040)
    batch_blur_repeat, _ = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats,
        instance_noise=blur_cfg,
        generator_step=12,
        batch_role="fake",
    )

    delta_xy = (batch_blur.x[:, :2] - batch_base.x[:, :2]).abs()
    assert torch.any(delta_xy > 0.0), "Blur enabled but X/Y features did not change."
    assert torch.equal(batch_base.x[:, 2], batch_blur.x[:, 2]), "Root marker changed."
    assert torch.equal(idx_base, idx_blur)
    assert torch.equal(batch_blur.x, batch_blur_repeat.x), "Fixed-seed blur is not deterministic."


def test_instance_noise_is_applied_pre_normalization() -> None:
    """Raw noise scales with observed sigma, normalized noise stays at tau."""
    torch.manual_seed(777)
    n = 4000
    edge_index = _path_graph_edge_index(n)
    roots = torch.tensor([0], dtype=torch.long)
    ego_cache = {
        0: (
            torch.arange(n, dtype=torch.long),
            edge_index,
            0,
        )
    }

    X = torch.zeros(n, dtype=torch.float32)
    Y = torch.zeros(n, dtype=torch.float32)

    blur_cfg = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.0,
        tau_y0=0.12,
        schedule="constant",
        anneal_steps=0,
        min_tau=0.0,
        apply_to="both",
    )
    tau_x, tau_y = compute_instance_noise_taus(blur_cfg, generator_step=50)
    assert tau_x == 0.0
    assert tau_y == 0.12

    norm_stats_small = {"mu_X": 0.0, "sigma_X": 1.0, "mu_Y": 0.0, "sigma_Y": 2.0}
    norm_stats_large = {"mu_X": 0.0, "sigma_X": 1.0, "mu_Y": 0.0, "sigma_Y": 20.0}

    torch.manual_seed(9191)
    batch_small, _ = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats_small,
        instance_noise=blur_cfg,
        generator_step=50,
        batch_role="real",
    )
    torch.manual_seed(9191)
    batch_large, _ = extract_ego_batch(
        roots=roots,
        ego_cache=ego_cache,
        X=X,
        Y=Y,
        norm_stats=norm_stats_large,
        instance_noise=blur_cfg,
        generator_step=50,
        batch_role="real",
    )

    y_tilde_small = batch_small.x[:, 1]
    y_tilde_large = batch_large.x[:, 1]
    assert torch.allclose(y_tilde_small, y_tilde_large, atol=1e-6, rtol=0.0)

    std_small = float(y_tilde_small.std(unbiased=False).item())
    std_large = float(y_tilde_large.std(unbiased=False).item())
    assert abs(std_small - tau_y) < 0.01
    assert abs(std_large - tau_y) < 0.01

    y_raw_small = y_tilde_small * norm_stats_small["sigma_Y"]
    y_raw_large = y_tilde_large * norm_stats_large["sigma_Y"]
    raw_ratio = float(y_raw_large.std(unbiased=False).item()) / float(
        y_raw_small.std(unbiased=False).item()
    )
    expected_ratio = norm_stats_large["sigma_Y"] / norm_stats_small["sigma_Y"]
    assert abs(raw_ratio - expected_ratio) < 0.1
    assert torch.equal(batch_small.x[:, 2], batch_large.x[:, 2])


def test_compute_instance_noise_taus_schedule_variants() -> None:
    """Scheduler returns expected tau values for constant/linear/exp modes."""
    cfg_constant = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.03,
        tau_y0=0.08,
        schedule="constant",
        anneal_steps=200,
        min_tau=0.0,
    )
    tx_c, ty_c = compute_instance_noise_taus(cfg_constant, generator_step=123)
    assert tx_c == 0.03
    assert ty_c == 0.08

    cfg_linear = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.10,
        tau_y0=0.20,
        schedule="linear",
        anneal_steps=100,
        min_tau=0.02,
    )
    tx_0, ty_0 = compute_instance_noise_taus(cfg_linear, generator_step=0)
    tx_mid, ty_mid = compute_instance_noise_taus(cfg_linear, generator_step=50)
    tx_end, ty_end = compute_instance_noise_taus(cfg_linear, generator_step=1000)
    assert tx_0 == 0.10 and ty_0 == 0.20
    assert abs(tx_mid - 0.05) < 1e-12
    assert abs(ty_mid - 0.10) < 1e-12
    assert tx_end == 0.02 and ty_end == 0.02

    cfg_exp = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.09,
        tau_y0=0.15,
        schedule="exp",
        anneal_steps=100,
        min_tau=0.01,
    )
    tx_e, ty_e = compute_instance_noise_taus(cfg_exp, generator_step=20)
    expected_tx = max(0.01, 0.09 * math.exp(-20.0 / (100.0 / 5.0)))
    expected_ty = max(0.01, 0.15 * math.exp(-20.0 / (100.0 / 5.0)))
    assert abs(tx_e - expected_tx) < 1e-12
    assert abs(ty_e - expected_ty) < 1e-12

