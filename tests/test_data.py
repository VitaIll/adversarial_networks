"""Tests for the NetworkData container: construction, validation, sanitisation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from adversarial_networks.data import NetworkData
from adversarial_networks.ego import EgoSubstrate
from adversarial_networks.generators import LinearInMeansGenerator
from adversarial_networks.sampling import RootSampler


def _clean_edge_index(n: int) -> torch.Tensor:
    src = torch.arange(n - 1, dtype=torch.long)
    dst = torch.arange(1, n, dtype=torch.long)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def test_from_edge_index_builds_and_exposes_properties() -> None:
    n = 8
    ei = _clean_edge_index(n)
    X = torch.randn(n)
    y = torch.randn(n)
    data = NetworkData.from_edge_index(ei, X, y, k=2)
    assert data.num_nodes == n
    assert data.k == 2
    assert torch.equal(data.y, y)
    assert torch.equal(data.X, X)
    assert isinstance(data.topology, EgoSubstrate)


def test_rejects_non_float32_outcome() -> None:
    n = 6
    ei = _clean_edge_index(n)
    X = torch.randn(n)
    with pytest.raises(TypeError, match="float32"):
        NetworkData.from_edge_index(ei, X, torch.randn(n, dtype=torch.float64), k=1)


def test_rejects_non_finite_outcome() -> None:
    n = 6
    ei = _clean_edge_index(n)
    X = torch.randn(n)
    y = torch.randn(n)
    y[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        NetworkData.from_edge_index(ei, X, y, k=1)


def test_rejects_wrong_length_outcome() -> None:
    n = 6
    ei = _clean_edge_index(n)
    X = torch.randn(n)
    with pytest.raises(ValueError):
        NetworkData.from_edge_index(ei, X, torch.randn(n + 2), k=1)


def test_validate_before_assign_leaves_no_half_built_object() -> None:
    n = 5
    sampler = RootSampler(num_nodes=n, mode="uniform", rng=np.random.default_rng(0))
    substrate = EgoSubstrate.from_edge_index(_clean_edge_index(n), torch.randn(n), k=1, root_sampler=sampler)
    with pytest.raises(ValueError):
        NetworkData(substrate, torch.randn(n + 1))  # wrong length -> raises before assigning


def test_from_networkx_reindexes_outcome_with_sanitisation() -> None:
    graph = nx.path_graph(6)
    graph.add_edge(2, 2)  # self-loop removed
    graph.add_node(99)    # isolated node dropped with the non-giant component
    X = torch.arange(7, dtype=torch.float32)  # aligned to sorted(nodes) == [0..5, 99]
    y = torch.arange(7, dtype=torch.float32) * 10.0
    data = NetworkData.from_networkx(graph, X, y, k=2)
    assert data.num_nodes == 6
    assert data.sanitization_report["self_loops_removed"] == 1
    assert data.sanitization_report["nodes_dropped"] == 1
    # y is re-indexed identically to X: the dropped node's value (60.0) is gone.
    assert torch.allclose(data.y, torch.arange(6, dtype=torch.float32) * 10.0)
    assert torch.allclose(data.X, torch.arange(6, dtype=torch.float32))


def test_simulate_classmethod_builds_outcome_from_model() -> None:
    graph = nx.barabasi_albert_graph(40, 2, seed=0)
    torch.manual_seed(0)
    X = torch.randn(graph.number_of_nodes())
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    data = NetworkData.simulate(graph, X, model, k=2, seed=1)
    # A Barabasi-Albert graph is connected and self-loop-free, so sanitisation drops
    # nothing: every node survives and the simulated outcome covers them all.
    assert data.num_nodes == graph.number_of_nodes()
    assert data.y.shape == (data.num_nodes,)
    assert data.y.dtype == torch.float32 and bool(torch.isfinite(data.y).all())


def test_to_networkx_roundtrips_node_count() -> None:
    data = NetworkData.from_edge_index(_clean_edge_index(7), torch.randn(7), torch.randn(7), k=1)
    g = data.to_networkx()
    assert g.number_of_nodes() == 7
