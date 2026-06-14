"""Deterministic CPU tests for ``core.graph.row_stochastic_weights``.

Relocated from the retired ``test_utils.py`` (the row-stochastic property check),
now exercising the kernel under its real name.
"""

from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.utils import from_networkx, to_undirected

from adversarial_networks.core.graph import row_stochastic_weights


def _path_graph_edge_index(num_nodes: int) -> torch.Tensor:
    """Create edge_index for a path graph."""
    graph = nx.path_graph(num_nodes)
    data = from_networkx(graph)
    return to_undirected(data.edge_index, num_nodes=num_nodes)


def test_W_row_stochastic() -> None:
    """W matrix is row-stochastic with proper shape and finite values."""
    n = 10
    edge_index = _path_graph_edge_index(n)
    W = row_stochastic_weights(edge_index=edge_index, num_nodes=n)
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
