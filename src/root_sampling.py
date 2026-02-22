"""Configurable root samplers for ego-graph batching.

The sampler is responsible only for selecting root node ids. Downstream ego
extraction and discriminator/generator logic remain unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from torch import Tensor

RootSamplerMode = Literal[
    "uniform",
    "disjoint_once",
    "disjoint_best_of_k",
    "disjoint_relax",
]
DisjointFallback = Literal["uniform", "best", "raise"]


@dataclass(frozen=True)
class RootSamplingResult:
    """Result payload for one sampling call."""

    roots: np.ndarray
    requested_size: int
    achieved_size: int
    mode: str
    attempts_used: int
    radius_used: int | None
    met_target: bool
    fallback_reason: str = ""


def build_adjacency_from_edge_index(edge_index: Tensor, num_nodes: int) -> list[np.ndarray]:
    """Build undirected adjacency lists from a PyG-style edge index.

    Args:
        edge_index: Long tensor with shape ``(2, num_edges)``.
        num_nodes: Number of nodes in ``[0, num_nodes)``.

    Returns:
        ``list[np.ndarray[int32]]`` with one neighbor array per node.
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
    """Validate and normalize adjacency representation to ``np.ndarray[int32]``."""
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


def precompute_balls(
    adjacency: Sequence[np.ndarray],
    radii: Iterable[int],
) -> dict[int, list[np.ndarray]]:
    """Precompute closed distance balls for one or more truncation radii."""
    num_nodes = len(adjacency)
    radii_sorted = sorted({int(radius) for radius in radii})
    if not radii_sorted:
        return {}

    balls_by_radius: dict[int, list[np.ndarray]] = {}
    for radius in radii_sorted:
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}.")
        balls_for_radius: list[np.ndarray] = []
        for start in range(num_nodes):
            visited = np.zeros(num_nodes, dtype=np.bool_)
            visited[start] = True
            queue: deque[tuple[int, int]] = deque([(start, 0)])

            while queue:
                node, dist = queue.popleft()
                if dist >= radius:
                    continue
                for nbr in adjacency[node]:
                    nbr_int = int(nbr)
                    if not visited[nbr_int]:
                        visited[nbr_int] = True
                        queue.append((nbr_int, dist + 1))

            balls_for_radius.append(np.flatnonzero(visited).astype(np.int32, copy=False))
        balls_by_radius[radius] = balls_for_radius

    return balls_by_radius


def greedy_pack_once_from_permutation(
    target_size: int,
    balls: Sequence[np.ndarray],
    permutation: np.ndarray,
) -> np.ndarray:
    """Run one greedy maximal-packing pass using a fixed node permutation."""
    num_nodes = len(balls)
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}.")
    if permutation.ndim != 1 or permutation.size != num_nodes:
        raise ValueError(
            "permutation must be a 1D array with exactly one entry per node."
        )
    if permutation.size > 0 and (int(permutation.min()) < 0 or int(permutation.max()) >= num_nodes):
        raise ValueError("permutation contains invalid node ids.")

    excluded = np.zeros(num_nodes, dtype=np.bool_)
    selected = np.empty(min(target_size, num_nodes), dtype=np.int64)
    count = 0

    for node_raw in permutation:
        node = int(node_raw)
        if excluded[node]:
            continue
        selected[count] = node
        count += 1
        excluded[balls[node]] = True
        if count >= target_size:
            break

    return selected[:count]


def greedy_pack_once(
    target_size: int,
    balls: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one random-order greedy maximal-packing pass."""
    permutation = rng.permutation(len(balls))
    return greedy_pack_once_from_permutation(
        target_size=target_size,
        balls=balls,
        permutation=permutation,
    )


def greedy_pack_best_from_permutations(
    target_size: int,
    balls: Sequence[np.ndarray],
    permutations: Sequence[np.ndarray],
) -> tuple[np.ndarray, int]:
    """Evaluate multiple greedy passes and return the largest selected root set."""
    if not permutations:
        raise ValueError("permutations must contain at least one permutation.")

    best = np.empty(0, dtype=np.int64)
    attempts_used = 0
    for permutation in permutations:
        attempts_used += 1
        candidate = greedy_pack_once_from_permutation(
            target_size=target_size,
            balls=balls,
            permutation=permutation,
        )
        if candidate.size > best.size:
            best = candidate
        if best.size >= target_size:
            break
    return best, attempts_used


def greedy_pack_best_of_k(
    target_size: int,
    balls: Sequence[np.ndarray],
    rng: np.random.Generator,
    restarts_k: int,
) -> tuple[np.ndarray, int]:
    """Run best-of-k greedy packing with early exit if target size is reached."""
    if restarts_k <= 0:
        raise ValueError(f"restarts_k must be positive, got {restarts_k}.")

    best = np.empty(0, dtype=np.int64)
    attempts_used = 0
    for _ in range(restarts_k):
        attempts_used += 1
        permutation = rng.permutation(len(balls))
        candidate = greedy_pack_once_from_permutation(
            target_size=target_size,
            balls=balls,
            permutation=permutation,
        )
        if candidate.size > best.size:
            best = candidate
        if best.size >= target_size:
            break
    return best, attempts_used


class RootSampler:
    """Single configurable entrypoint for root selection."""

    def __init__(
        self,
        num_nodes: int,
        mode: RootSamplerMode = "uniform",
        exclusion_r: int = 3,
        disjoint_restarts_k: int = 3,
        disjoint_min_batch: int | None = None,
        disjoint_relax_sequence: Sequence[int] = (3, 2),
        disjoint_fallback: DisjointFallback = "uniform",
        rng: np.random.Generator | None = None,
        adjacency: Sequence[Sequence[int] | np.ndarray] | None = None,
    ) -> None:
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}.")
        if mode not in {"uniform", "disjoint_once", "disjoint_best_of_k", "disjoint_relax"}:
            raise ValueError(f"Unknown sampler mode: {mode!r}.")
        if exclusion_r < 0:
            raise ValueError(f"exclusion_r must be non-negative, got {exclusion_r}.")
        if disjoint_restarts_k <= 0:
            raise ValueError(
                f"disjoint_restarts_k must be positive, got {disjoint_restarts_k}."
            )
        if disjoint_min_batch is not None and disjoint_min_batch <= 0:
            raise ValueError(
                "disjoint_min_batch must be positive when provided, "
                f"got {disjoint_min_batch}."
            )
        if disjoint_fallback not in {"uniform", "best", "raise"}:
            raise ValueError(f"Unknown disjoint_fallback: {disjoint_fallback!r}.")

        relax_seq = tuple(int(radius) for radius in disjoint_relax_sequence)
        if mode == "disjoint_relax" and not relax_seq:
            raise ValueError("disjoint_relax_sequence must be non-empty in disjoint_relax mode.")
        if any(radius < 0 for radius in relax_seq):
            raise ValueError("disjoint_relax_sequence entries must be non-negative.")

        self.num_nodes = int(num_nodes)
        self.mode = mode
        self.exclusion_r = int(exclusion_r)
        self.disjoint_restarts_k = int(disjoint_restarts_k)
        self.disjoint_min_batch = (
            int(disjoint_min_batch) if disjoint_min_batch is not None else None
        )
        self.disjoint_relax_sequence = relax_seq
        self.disjoint_fallback = disjoint_fallback
        self.rng = rng if rng is not None else np.random.default_rng()

        self.adjacency: list[np.ndarray] | None = None
        self.balls_by_radius: dict[int, list[np.ndarray]] = {}
        if mode != "uniform":
            if adjacency is None:
                raise ValueError("adjacency is required for disjoint sampler modes.")
            self.adjacency = normalize_adjacency(adjacency=adjacency, num_nodes=self.num_nodes)
            self.balls_by_radius = precompute_balls(
                adjacency=self.adjacency,
                radii=self._required_radii(),
            )

    def _required_radii(self) -> tuple[int, ...]:
        if self.mode in {"disjoint_once", "disjoint_best_of_k"}:
            return (self.exclusion_r,)
        if self.mode == "disjoint_relax":
            return self.disjoint_relax_sequence
        return ()

    def _uniform_result(self, batch_size: int, fallback_reason: str = "") -> RootSamplingResult:
        roots = self.rng.integers(
            low=0,
            high=self.num_nodes,
            size=batch_size,
            dtype=np.int64,
        )
        return RootSamplingResult(
            roots=roots,
            requested_size=batch_size,
            achieved_size=int(roots.size),
            mode="uniform",
            attempts_used=1,
            radius_used=None,
            met_target=True,
            fallback_reason=fallback_reason,
        )

    def _disjoint_once_result(self, batch_size: int, radius: int) -> RootSamplingResult:
        roots = greedy_pack_once(
            target_size=batch_size,
            balls=self.balls_by_radius[radius],
            rng=self.rng,
        )
        return RootSamplingResult(
            roots=roots,
            requested_size=batch_size,
            achieved_size=int(roots.size),
            mode=self.mode,
            attempts_used=1,
            radius_used=radius,
            met_target=bool(roots.size >= batch_size),
        )

    def _disjoint_best_of_k_result(self, batch_size: int, radius: int) -> RootSamplingResult:
        roots, attempts_used = greedy_pack_best_of_k(
            target_size=batch_size,
            balls=self.balls_by_radius[radius],
            rng=self.rng,
            restarts_k=self.disjoint_restarts_k,
        )
        return RootSamplingResult(
            roots=roots,
            requested_size=batch_size,
            achieved_size=int(roots.size),
            mode="disjoint_best_of_k",
            attempts_used=attempts_used,
            radius_used=radius,
            met_target=bool(roots.size >= batch_size),
        )

    def _disjoint_relax_result(self, batch_size: int) -> RootSamplingResult:
        min_batch = int(self.disjoint_min_batch or batch_size)
        min_batch = min(min_batch, batch_size)

        total_attempts = 0
        best_result: RootSamplingResult | None = None
        for radius in self.disjoint_relax_sequence:
            roots, attempts_used = greedy_pack_best_of_k(
                target_size=batch_size,
                balls=self.balls_by_radius[radius],
                rng=self.rng,
                restarts_k=self.disjoint_restarts_k,
            )
            total_attempts += attempts_used
            candidate = RootSamplingResult(
                roots=roots,
                requested_size=batch_size,
                achieved_size=int(roots.size),
                mode="disjoint_relax",
                attempts_used=total_attempts,
                radius_used=radius,
                met_target=bool(roots.size >= batch_size),
            )
            if best_result is None or candidate.achieved_size > best_result.achieved_size:
                best_result = candidate
            if candidate.achieved_size >= min_batch:
                return candidate

        assert best_result is not None
        if self.disjoint_fallback == "best":
            return RootSamplingResult(
                roots=best_result.roots,
                requested_size=batch_size,
                achieved_size=best_result.achieved_size,
                mode="disjoint_relax",
                attempts_used=best_result.attempts_used,
                radius_used=best_result.radius_used,
                met_target=best_result.met_target,
                fallback_reason=f"best_below_min_batch_{min_batch}",
            )
        if self.disjoint_fallback == "uniform":
            fallback = self._uniform_result(
                batch_size=batch_size,
                fallback_reason=f"uniform_fallback_below_min_batch_{min_batch}",
            )
            return RootSamplingResult(
                roots=fallback.roots,
                requested_size=fallback.requested_size,
                achieved_size=fallback.achieved_size,
                mode=fallback.mode,
                attempts_used=total_attempts + fallback.attempts_used,
                radius_used=fallback.radius_used,
                met_target=fallback.met_target,
                fallback_reason=fallback.fallback_reason,
            )

        raise RuntimeError(
            "disjoint_relax failed to achieve minimum packed roots "
            f"(min_batch={min_batch}, best={best_result.achieved_size})."
        )

    def sample(self, batch_size: int) -> RootSamplingResult:
        """Sample root ids for one discriminator or generator batch."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        if self.mode == "uniform":
            return self._uniform_result(batch_size=batch_size)
        if self.mode == "disjoint_once":
            return self._disjoint_once_result(batch_size=batch_size, radius=self.exclusion_r)
        if self.mode == "disjoint_best_of_k":
            return self._disjoint_best_of_k_result(
                batch_size=batch_size,
                radius=self.exclusion_r,
            )
        return self._disjoint_relax_result(batch_size=batch_size)


def sample_roots_tensor(
    sampler: RootSampler,
    batch_size: int,
    device: torch.device | str,
) -> tuple[Tensor, RootSamplingResult]:
    """Sample roots and return both tensor and metadata."""
    result = sampler.sample(batch_size=batch_size)
    roots_tensor = torch.as_tensor(result.roots, dtype=torch.long, device=device)
    return roots_tensor, result
