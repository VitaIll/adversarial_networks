"""Fast computational core — pure, dependency-light numeric primitives.

The kernels (``equilibrium``, ``graph``, ``neighborhoods``, ``objective``) import
no ``torch_geometric``; the single documented ``core`` ↔ PyG seam is
``ego_features.extract_ego_batch`` (the paper's ego-extraction primitive). Workflow
modules import *from* here, never the reverse.
"""

from __future__ import annotations

from .ego_features import EgoCache, EgoCacheEntry, NormStats, extract_ego_batch
from .equilibrium import newton, picard, solve_equilibrium
from .graph import (
    adjacency_lists_from_edge_index,
    normalize_adjacency,
    row_stochastic_weights,
)
from .neighborhoods import (
    greedy_pack_best_from_permutations,
    greedy_pack_best_of_k,
    greedy_pack_once,
    greedy_pack_once_from_permutation,
    precompute_balls,
)
from .objective import (
    OPTIMAL_DISC_LOSS,
    OPTIMAL_GEN_LOSS,
    check_gan_convergence,
    discriminator_loss,
    generator_nonsaturating_loss,
    generator_saturating_loss,
    instance_noise_taus,
)
from .types import InstanceNoiseConfigLike

__all__ = [
    # equilibrium
    "picard",
    "newton",
    "solve_equilibrium",
    # graph
    "row_stochastic_weights",
    "adjacency_lists_from_edge_index",
    "normalize_adjacency",
    # neighborhoods
    "precompute_balls",
    "greedy_pack_once",
    "greedy_pack_once_from_permutation",
    "greedy_pack_best_of_k",
    "greedy_pack_best_from_permutations",
    # ego features (the PyG seam)
    "extract_ego_batch",
    "EgoCache",
    "EgoCacheEntry",
    "NormStats",
    # objective
    "discriminator_loss",
    "generator_nonsaturating_loss",
    "generator_saturating_loss",
    "check_gan_convergence",
    "instance_noise_taus",
    "OPTIMAL_DISC_LOSS",
    "OPTIMAL_GEN_LOSS",
    # types
    "InstanceNoiseConfigLike",
]
