"""Tests for the EgoSubstrate data object.

Covers: consistent construction from a clean edge index (row-stochastic W, full
ego-cache coverage), root sampling and rooted-ego batch construction (feature
layout and root markers), normalisation-stat computation and its guards, boundary
validation, and NetworkX sanitisation with covariate re-indexing.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from src.ego import EgoSubstrate
from src.root_sampling import RootSampler


def _path_edge_index(n: int) -> torch.Tensor:
    src = torch.arange(n - 1, dtype=torch.long)
    dst = torch.arange(1, n, dtype=torch.long)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def _uniform_substrate(n: int, k: int, seed: int = 0) -> EgoSubstrate:
    edge_index = _path_edge_index(n)
    X = torch.randn(n)
    sampler = RootSampler(num_nodes=n, mode="uniform", rng=np.random.default_rng(seed))
    return EgoSubstrate.from_edge_index(edge_index, X, k=k, root_sampler=sampler)


def test_from_edge_index_builds_row_stochastic_substrate() -> None:
    n = 6
    sub = _uniform_substrate(n, k=2)
    assert sub.num_nodes == n
    assert sub.W.shape == (n, n) and sub.W.is_sparse
    assert len(sub.ego_cache) == n
    assert sub.sigma_X > 0.0
    rowsums = torch.sparse.sum(sub.W, dim=1).to_dense()
    assert torch.allclose(rowsums, torch.ones(n), atol=1e-6)


def test_sample_roots_returns_long_tensor_on_device() -> None:
    sub = _uniform_substrate(8, k=2, seed=1)
    roots, result = sub.sample_roots(4)
    assert roots.dtype == torch.long
    assert roots.numel() == 4
    assert result.requested_size == 4
    assert roots.device == sub.device


def test_build_batch_feature_layout_and_root_markers() -> None:
    n = 8
    sub = _uniform_substrate(n, k=2, seed=2)
    Y = torch.randn(n)
    norm = sub.make_norm_stats(Y)
    roots, _ = sub.sample_roots(4)
    batch, root_idx = sub.build_batch(roots, Y, norm, step=0, role="real")
    assert batch.x.shape[1] == 3  # [X_tilde, Y_tilde, root_marker]
    assert root_idx.numel() == 4
    # The root marker (column 2) must be exactly 1 at each root position.
    assert torch.allclose(batch.x[root_idx, 2], torch.ones(4))
    # Non-root nodes must have a zero root marker.
    marker_sum = batch.x[:, 2].sum().item()
    assert abs(marker_sum - 4.0) < 1e-6


def test_make_norm_stats_combines_fixed_X_and_outcome_stats() -> None:
    sub = _uniform_substrate(7, k=1, seed=3)
    Y = torch.randn(7)
    norm = sub.make_norm_stats(Y)
    assert set(norm.keys()) == {"mu_X", "sigma_X", "mu_Y", "sigma_Y"}
    assert norm["mu_X"] == pytest.approx(sub.mu_X)
    assert norm["sigma_X"] == pytest.approx(sub.sigma_X)
    assert norm["sigma_Y"] > 0.0


def test_make_norm_stats_rejects_constant_outcome() -> None:
    sub = _uniform_substrate(5, k=1, seed=4)
    with pytest.raises(ValueError):
        sub.make_norm_stats(torch.ones(5))


def test_constructor_rejects_sampler_node_mismatch() -> None:
    n = 5
    edge_index = _path_edge_index(n)
    X = torch.randn(n)
    sampler = RootSampler(num_nodes=n + 1, mode="uniform", rng=np.random.default_rng(5))
    with pytest.raises(ValueError):
        EgoSubstrate.from_edge_index(edge_index, X, k=1, root_sampler=sampler)


def test_from_networkx_sanitizes_selfloops_components_and_reindexes_X() -> None:
    graph = nx.path_graph(6)  # nodes 0..5 connected in a path
    graph.add_edge(2, 2)  # a self-loop to be removed
    graph.add_node(99)  # an isolated node, dropped with the non-giant component
    # X aligned to sorted(nodes) == [0,1,2,3,4,5,99]; node 99 carries value 6.
    X = torch.arange(7, dtype=torch.float32)

    sub = EgoSubstrate.from_networkx(graph, X, k=2)

    assert sub.num_nodes == 6
    assert sub.sanitization_report["self_loops_removed"] == 1
    assert sub.sanitization_report["nodes_dropped"] == 1
    # The dropped node's covariate (value 6) must be gone; survivors keep 0..5.
    assert torch.allclose(sub.X, torch.arange(6, dtype=torch.float32))
    rowsums = torch.sparse.sum(sub.W, dim=1).to_dense()
    assert torch.allclose(rowsums, torch.ones(6), atol=1e-6)


def test_from_networkx_rejects_mismatched_covariate_length() -> None:
    graph = nx.path_graph(4)
    with pytest.raises(ValueError):
        EgoSubstrate.from_networkx(graph, torch.randn(3), k=1)
