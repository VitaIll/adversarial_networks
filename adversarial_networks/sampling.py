"""Configurable root samplers for ego-graph batching.

The sampler selects root node ids for a discriminator/generator batch — either a
plain uniform draw, or a *disjoint* greedy packing that excludes, around each
accepted root ``u``, the closed ball of radius ``exclusion_r`` (so the selected
roots satisfy ``dist(u, v) > exclusion_r`` pairwise, i.e. their radius-``exclusion_r``
balls are disjoint by construction). This is the paper's batch-overlap fix
(Illichmann & Zacchia, 2026, Sec. 4.2, fn. 26): two radius-``k`` ego balls are
*vertex-disjoint* iff the distance between their centres exceeds ``2k``, so the
sampled radius-``k`` ego objects are vertex-disjoint — hence near-independent — only
when ``exclusion_r >= 2k`` (with ``k`` the discriminator/ego depth). The sampler
itself does not know ``k``; choosing/validating ``exclusion_r = 2k`` is the job of
the caller that owns ``k`` (see :class:`adversarial_networks.ego.EgoSubstrate`).
The graph/packing primitives live in :mod:`adversarial_networks.core.graph` and
:mod:`adversarial_networks.core.neighborhoods`; this module is the configured
front end over them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .core.graph import normalize_adjacency
from .core.neighborhoods import greedy_pack_best_of_k, greedy_pack_once, precompute_balls

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
    """Sample roots and return both the long tensor and the sampler metadata."""
    result = sampler.sample(batch_size=batch_size)
    roots_tensor = torch.as_tensor(result.roots, dtype=torch.long, device=device)
    return roots_tensor, result
