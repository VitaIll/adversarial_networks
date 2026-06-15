"""Disjoint-neighbourhood packing primitives (pure ``numpy``).

These support the disjoint root samplers: precompute closed distance balls once,
then greedily pack a maximal set of roots whose *exclusion* balls (the radius
passed in, ``exclusion_r``) are disjoint by construction — i.e. every pair of
selected roots ``u, v`` satisfies ``dist(u, v) > exclusion_r``. The packer is
radius-agnostic: it excludes whatever ``balls[node]`` it is handed. Disjoint
exclusion balls make the sampled radius-``k`` *ego objects* vertex-disjoint (hence
near-independent) only when ``exclusion_r >= 2k``, since two radius-``k`` balls are
vertex-disjoint iff the distance between their centres exceeds ``2k`` (Illichmann &
Zacchia, 2026, Sec. 4.2, fn. 26). Choosing that radius is the caller's job. Used
by :mod:`adversarial_networks.sampling`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np


def precompute_balls(
    adjacency: Sequence[np.ndarray],
    radii: Iterable[int],
) -> dict[int, list[np.ndarray]]:
    """Precompute closed distance balls for one or more truncation radii.

    Args:
        adjacency: One neighbour array per node (normalised int32).
        radii: Non-negative truncation radii to materialise.

    Returns:
        ``{radius: [ball_node0, ball_node1, ...]}`` where each ball is the int32
        array of node ids within ``radius`` hops (inclusive).
    """
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
