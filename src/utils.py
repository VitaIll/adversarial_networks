"""Core utility functions for the linear-in-means adversarial estimation MVP.

This module provides essential utility functions for:
    - Building row-stochastic network weight matrices from edge lists
    - Extracting and batching rooted ego-subgraphs with normalized features

The GAN component classes (SCMGenerator and RootedMPNNDiscriminator) have been
moved to dedicated modules for better organization:
    - src.generator.SCMGenerator: Structural causal model generator
    - src.discriminator.RootedMPNNDiscriminator: Root-aware MPNN discriminator
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree

EgoCacheEntry: TypeAlias = tuple[Tensor, Tensor, int]
EgoCache: TypeAlias = Mapping[int, EgoCacheEntry]
NormStats: TypeAlias = Mapping[str, float]


class InstanceNoiseConfigLike(Protocol):
    """Structural type for optional instance-noise configuration."""

    enabled: bool
    tau_x0: float
    tau_y0: float
    schedule: str
    anneal_steps: int
    min_tau: float
    apply_to: str


def build_row_stochastic_W(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Build a sparse row-stochastic network weight matrix.

    Args:
        edge_index: Long tensor of shape `(2, num_edges)` on CPU/GPU with graph
            edges. For undirected graphs, both directions must be present.
        num_nodes: Positive number of nodes `n`.

    Returns:
        A coalesced `torch.sparse_coo_tensor` `W` of shape `(n, n)`, `float32`,
        on the same device as `edge_index`. For each node with nonzero degree,
        row sums satisfy `sum_j W[i, j] = 1.0`.

    Raises:
        TypeError: If `edge_index` does not have integer dtype.
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
    row_deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
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


def compute_instance_noise_taus(
    instance_noise: InstanceNoiseConfigLike | None,
    generator_step: int,
) -> tuple[float, float]:
    """Compute scheduled blur intensity (normalized units) for a generator step.

    Args:
        instance_noise: Instance-noise configuration. If `None` or disabled,
            returns `(0.0, 0.0)`.
        generator_step: Outer generator step counter (0-based or 1-based).

    Returns:
        `(tau_x, tau_y)` in normalized units.
    """
    if instance_noise is None or not bool(instance_noise.enabled):
        return 0.0, 0.0

    step = max(0, int(generator_step))
    schedule = str(instance_noise.schedule)
    anneal_steps = int(instance_noise.anneal_steps)
    min_tau = float(instance_noise.min_tau)

    if schedule not in {"constant", "linear", "exp"}:
        raise ValueError(
            "instance-noise schedule must be one of {'constant', 'linear', 'exp'}, "
            f"got {schedule!r}."
        )
    if anneal_steps < 0:
        raise ValueError(f"anneal_steps must be non-negative, got {anneal_steps}.")
    if min_tau < 0.0:
        raise ValueError(f"min_tau must be non-negative, got {min_tau}.")

    def _tau_at_step(tau0: float) -> float:
        tau0 = float(tau0)
        if tau0 < 0.0:
            raise ValueError(f"tau0 must be non-negative, got {tau0}.")
        if schedule == "constant" or anneal_steps == 0:
            return max(min_tau, tau0)
        if schedule == "linear":
            if step <= anneal_steps:
                frac = 1.0 - (float(step) / float(anneal_steps))
                return max(min_tau, tau0 * frac)
            return max(min_tau, min_tau)
        # Exponential annealing with decay derived from anneal_steps.
        tau_decay = max(1.0, float(anneal_steps) / 5.0)
        return max(min_tau, tau0 * math.exp(-float(step) / tau_decay))

    tau_x = _tau_at_step(float(instance_noise.tau_x0))
    tau_y = _tau_at_step(float(instance_noise.tau_y0))
    return tau_x, tau_y


def extract_ego_batch(
    roots: Tensor | Sequence[int],
    ego_cache: EgoCache,
    X: Tensor,
    Y: Tensor,
    norm_stats: NormStats,
    instance_noise: InstanceNoiseConfigLike | None = None,
    generator_step: int = 0,
    batch_role: Literal["real", "fake"] = "fake",
) -> tuple[Batch, Tensor]:
    """Build a PyG batch of rooted ego subgraphs with normalized features.

    Args:
        roots: Root node ids, shape `(batch_size,)`, integer tensor or sequence.
        ego_cache: Mapping `root -> (subset, sub_edge_index, root_pos)` where:
            `subset` is long `(num_sub_nodes,)`,
            `sub_edge_index` is long `(2, num_sub_edges)`,
            `root_pos` is int in `[0, num_sub_nodes)`.
        X: Node covariate tensor `(n,)`, float, same device as `Y`.
        Y: Node outcome tensor `(n,)`, float, same device as `X`.
        norm_stats: Frozen normalization stats with keys
            `mu_X`, `sigma_X`, `mu_Y`, `sigma_Y` as positive Python floats.
        instance_noise: Optional blur configuration applied to discriminator
            inputs before normalization.
        generator_step: Current outer generator step for blur schedule lookup.
        batch_role: Whether this is a real or fake discriminator batch.

    Returns:
        A tuple `(batch, root_indices)`:
        - `batch`: PyG `Batch` with concatenated node features
          shape `(sum_r |B_k(root_r)|, 3)`, float, columns
          `[X_tilde, Y_tilde, root_marker]`.
        - `root_indices`: Long tensor `(batch_size,)` indexing each root node in
          the concatenated batch node axis.

    Raises:
        KeyError: If normalization keys are missing.
        TypeError/ValueError: For invalid tensor shapes/dtypes/devices.
    """
    required = {"mu_X", "sigma_X", "mu_Y", "sigma_Y"}
    missing = required.difference(norm_stats.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"norm_stats missing required keys: {missing_str}.")

    if not isinstance(X, Tensor) or not isinstance(Y, Tensor):
        raise TypeError("X and Y must be torch.Tensor objects.")
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("X and Y must have shape (n,).")
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")
    if not torch.is_floating_point(X) or not torch.is_floating_point(Y):
        raise TypeError("X and Y must have floating dtypes.")
    if X.device != Y.device:
        raise ValueError("X and Y must be on the same device.")

    if isinstance(roots, Tensor):
        if roots.dtype != torch.long:
            raise TypeError("roots tensor must have dtype torch.long.")
        if roots.ndim != 1:
            raise ValueError("roots tensor must have shape (batch_size,).")
        root_list = [int(v) for v in roots.tolist()]
    else:
        root_list = [int(v) for v in roots]

    if not root_list:
        raise ValueError("roots cannot be empty.")
    if batch_role not in {"real", "fake"}:
        raise ValueError("batch_role must be 'real' or 'fake'.")

    mu_X = float(norm_stats["mu_X"])
    sigma_X = float(norm_stats["sigma_X"])
    mu_Y = float(norm_stats["mu_Y"])
    sigma_Y = float(norm_stats["sigma_Y"])
    if sigma_X <= 0.0 or sigma_Y <= 0.0:
        raise ValueError("sigma_X and sigma_Y must be strictly positive.")

    tau_x, tau_y = compute_instance_noise_taus(
        instance_noise=instance_noise,
        generator_step=generator_step,
    )
    add_blur = False
    if instance_noise is not None and bool(instance_noise.enabled):
        apply_to = str(instance_noise.apply_to)
        if apply_to == "real_only":
            warnings.warn(
                "instance_noise.apply_to='real_only' is non-default and distorts only "
                "real discriminator targets.",
                RuntimeWarning,
                stacklevel=2,
            )
        if apply_to not in {"both", "real_only"}:
            raise ValueError(
                "instance_noise.apply_to must be one of {'both', 'real_only'}, got "
                f"{apply_to!r}."
            )
        add_blur = apply_to == "both" or (apply_to == "real_only" and batch_role == "real")

    sigma_x_raw = tau_x * sigma_X
    sigma_y_raw = tau_y * sigma_Y

    data_list: list[Data] = []
    root_positions: list[int] = []
    for root in root_list:
        if root not in ego_cache:
            raise KeyError(f"Root {root} missing from ego_cache.")
        subset, sub_edge_index, root_pos = ego_cache[root]

        if subset.dtype != torch.long or subset.ndim != 1:
            raise TypeError("ego_cache subset must be a 1D torch.long tensor.")
        if sub_edge_index.dtype != torch.long or sub_edge_index.ndim != 2:
            raise TypeError("ego_cache sub_edge_index must be a 2D torch.long tensor.")
        if sub_edge_index.shape[0] != 2:
            raise ValueError("ego_cache sub_edge_index must have shape (2, num_edges).")
        if not (0 <= root_pos < subset.shape[0]):
            raise ValueError("ego_cache root_pos is out of range for subset size.")
        if int(subset.min().item()) < 0 or int(subset.max().item()) >= X.shape[0]:
            raise ValueError("ego_cache subset contains invalid node ids.")

        subset_local = subset.to(device=X.device)
        edge_local = sub_edge_index.to(device=X.device)

        X_sub = X.index_select(0, subset_local)
        Y_sub = Y.index_select(0, subset_local)
        if add_blur and sigma_x_raw > 0.0:
            X_sub = X_sub + torch.randn_like(X_sub) * sigma_x_raw
        if add_blur and sigma_y_raw > 0.0:
            Y_sub = Y_sub + torch.randn_like(Y_sub) * sigma_y_raw

        X_tilde = (X_sub - mu_X) / sigma_X
        Y_tilde = (Y_sub - mu_Y) / sigma_Y
        root_marker = torch.zeros_like(X_tilde)
        root_marker[root_pos] = 1.0
        features = torch.stack((X_tilde, Y_tilde, root_marker), dim=1)

        data_list.append(Data(x=features, edge_index=edge_local))
        root_positions.append(int(root_pos))

    batch = Batch.from_data_list(data_list)
    ptr = batch.ptr[:-1]
    root_offsets = torch.tensor(root_positions, dtype=torch.long, device=ptr.device)
    root_indices = ptr + root_offsets
    return batch, root_indices
