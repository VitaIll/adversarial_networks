"""Graph-algebra primitives: the row-stochastic interaction matrix and adjacency.

Pure ``torch``/``numpy`` (no ``torch_geometric``): the row degree is computed with
``torch.bincount`` rather than ``torch_geometric.utils.degree`` so this kernel —
the building block of every equilibrium solve — stays dependency-light and
independently testable. ``torch_geometric.utils.degree`` is itself a thin
``scatter_add`` of ones, i.e. ``bincount``; the two are value-identical for a
non-negative 1-D index.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def row_stochastic_weights(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Build the sparse row-stochastic interaction matrix ``W``.

    Args:
        edge_index: Long tensor of shape ``(2, num_edges)`` with graph edges. For
            undirected graphs, both directions must be present.
        num_nodes: Positive number of nodes ``n``.

    Returns:
        A coalesced ``torch.sparse_coo_tensor`` ``W`` of shape ``(n, n)``,
        ``float32``, on the same device as ``edge_index``. For each node with
        nonzero degree, the row sums satisfy ``sum_j W[i, j] = 1.0``; its nonzero
        entries are ``1/degree_i`` on the graph edges (so the *indices are the
        adjacency* and the *values encode the degree* — a general aggregate
        ``sum_j a_ij g(Y_j)`` is recoverable from ``W``).

    Raises:
        TypeError: If ``edge_index`` does not have ``torch.long`` dtype.
        ValueError: If shapes, node count, or degree constraints are invalid.
    """
    if not isinstance(edge_index, Tensor):
        raise TypeError("edge_index must be a torch.Tensor.")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if edge_index.numel() == 0:
        raise ValueError("edge_index is empty.")
    if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node ids outside [0, num_nodes).")

    row = edge_index[0]
    col = edge_index[1]
    row_deg = torch.bincount(row, minlength=num_nodes).to(torch.float32)
    if torch.any(row_deg <= 0):
        raise ValueError(
            "All nodes must have positive degree to build a row-stochastic matrix."
        )

    values = row_deg.reciprocal().index_select(0, row)
    W = torch.sparse_coo_tensor(
        indices=torch.stack((row, col), dim=0),
        values=values,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
        device=edge_index.device,
    )
    return W.coalesce()


def adjacency_lists_from_edge_index(edge_index: Tensor, num_nodes: int) -> list[np.ndarray]:
    """Build undirected adjacency lists from a PyG-style edge index.

    Args:
        edge_index: Long tensor with shape ``(2, num_edges)``.
        num_nodes: Number of nodes in ``[0, num_nodes)``.

    Returns:
        ``list[np.ndarray[int32]]`` with one sorted neighbour array per node.

    Raises:
        TypeError, ValueError: On dtype/shape/range violations.
    """
    if not isinstance(edge_index, Tensor):
        raise TypeError("edge_index must be a torch.Tensor.")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}.")
    if edge_index.numel() == 0:
        raise ValueError("edge_index cannot be empty.")

    edge_np = edge_index.detach().cpu().numpy()
    row = edge_np[0]
    col = edge_np[1]
    if int(row.min()) < 0 or int(col.min()) < 0:
        raise ValueError("edge_index contains negative node ids.")
    if int(row.max()) >= num_nodes or int(col.max()) >= num_nodes:
        raise ValueError("edge_index contains node ids >= num_nodes.")

    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for src_raw, dst_raw in zip(row, col, strict=True):
        src = int(src_raw)
        dst = int(dst_raw)
        if src == dst:
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)

    adjacency: list[np.ndarray] = []
    for node_neighbors in neighbors:
        if not node_neighbors:
            adjacency.append(np.empty(0, dtype=np.int32))
            continue
        adjacency.append(
            np.fromiter(sorted(node_neighbors), dtype=np.int32, count=len(node_neighbors))
        )
    return adjacency


def normalize_adjacency(
    adjacency: Sequence[Sequence[int] | np.ndarray],
    num_nodes: int,
) -> list[np.ndarray]:
    """Validate and normalise an adjacency representation to ``np.ndarray[int32]``.

    Drops self-loops and duplicate neighbours, sorts each list, and validates the
    node-id range.

    Raises:
        ValueError: On size or range violations.
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}.")
    if len(adjacency) != num_nodes:
        raise ValueError(
            f"adjacency must contain exactly num_nodes entries, got {len(adjacency)}."
        )

    normalized: list[np.ndarray] = []
    for node, neighbors in enumerate(adjacency):
        arr = np.asarray(neighbors, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError("Each adjacency entry must be a 1D sequence of neighbors.")
        if arr.size > 0 and (int(arr.min()) < 0 or int(arr.max()) >= num_nodes):
            raise ValueError("adjacency contains node ids outside [0, num_nodes).")
        arr = arr[arr != node]
        arr = np.unique(arr)
        normalized.append(arr.astype(np.int32, copy=False))
    return normalized
