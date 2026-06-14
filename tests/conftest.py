"""Shared pytest fixtures for all tests.

Provides reusable test fixtures for graphs, configurations, and common test data.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch
from torch_geometric.utils import from_networkx, k_hop_subgraph, to_undirected

from adversarial_networks.core.graph import row_stochastic_weights as build_row_stochastic_W


@pytest.fixture
def small_path_graph() -> tuple[torch.Tensor, int]:
    """Create a small path graph for testing.

    Returns:
        Tuple of (edge_index, num_nodes).
    """
    n = 10
    graph = nx.path_graph(n)
    data = from_networkx(graph)
    edge_index = to_undirected(data.edge_index, num_nodes=n)
    return edge_index, n


@pytest.fixture
def small_row_stochastic_W(small_path_graph) -> torch.Tensor:
    """Build row-stochastic weight matrix from small path graph.

    Returns:
        Sparse COO tensor W of shape (n, n).
    """
    edge_index, n = small_path_graph
    return build_row_stochastic_W(edge_index=edge_index, num_nodes=n)


@pytest.fixture
def norm_stats_fixture() -> dict[str, float]:
    """Standard normalization statistics for testing.

    Returns:
        Dictionary with mu_X, sigma_X, mu_Y, sigma_Y.
    """
    return {
        "mu_X": 0.0,
        "sigma_X": 1.0,
        "mu_Y": 0.0,
        "sigma_Y": 1.0,
    }


@pytest.fixture
def ego_cache_fixture(small_path_graph) -> dict[int, tuple[torch.Tensor, torch.Tensor, int]]:
    """Build ego-graph cache for small path graph with k=2.

    Returns:
        Dictionary mapping root -> (subset, sub_edge_index, root_pos).
    """
    edge_index, n = small_path_graph
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

    return ego_cache


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
