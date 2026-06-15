"""Rooted ego-batch construction — the single ``core`` ↔ PyTorch-Geometric seam.

:func:`extract_ego_batch` is the paper's computational primitive (ii): given a set
of root node ids and the precomputed ``k``-ego cache, it assembles a PyG ``Batch``
of rooted attributed ego-subgraphs with node features ``[X̃ (d_x cols), Ỹ, root_marker]``
and the per-root index into the concatenated node axis. The covariate ``X`` may be a
single scalar per node (shape ``(n,)``, ``d_x = 1``) or a vector per node (shape
``(n, d_x)``, ``d_x >= 1``); the outcome ``Y`` stays scalar per node. This is the
*only* core module that imports ``torch_geometric``; the numeric kernels
(``equilibrium``, ``graph``, ``neighborhoods``, ``objective``) stay PyG-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from .objective import instance_noise_taus
from .types import InstanceNoiseConfigLike

EgoCacheEntry: TypeAlias = tuple[Tensor, Tensor, int]
"""``(subset, sub_edge_index, root_pos)`` for one root's induced ``k``-ego subgraph."""
EgoCache: TypeAlias = Mapping[int, EgoCacheEntry]
"""Mapping ``root -> EgoCacheEntry`` covering every node."""
NormStats: TypeAlias = Mapping[str, "float | Sequence[float] | Tensor"]
"""Frozen normalisation stats ``{mu_X, sigma_X, mu_Y, sigma_Y}`` (positive scales).

``mu_Y``/``sigma_Y`` are always scalars (the outcome is scalar per node). For scalar
covariates (1-D ``X``) ``mu_X``/``sigma_X`` are scalars too; for vector covariates
(2-D ``X`` with ``d_x`` columns) they are per-column sequences/tensors of length ``d_x``.
"""


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
    """Build a PyG batch of rooted ego subgraphs with normalised features.

    Args:
        roots: Root node ids, shape ``(batch_size,)``, integer tensor or sequence.
        ego_cache: Mapping ``root -> (subset, sub_edge_index, root_pos)``.
        X: Node covariate tensor ``(n,)`` (scalar covariate, ``d_x = 1``) or
            ``(n, d_x)`` (vector covariate, ``d_x >= 1``), float, same device as ``Y``.
        Y: Node outcome tensor ``(n,)``, float, same device as ``X``.
        norm_stats: Frozen normalisation stats with keys ``mu_X``, ``sigma_X``,
            ``mu_Y``, ``sigma_Y``. ``mu_Y``/``sigma_Y`` are positive Python floats;
            ``mu_X``/``sigma_X`` are scalars for 1-D ``X`` and per-column sequences of
            length ``d_x`` for 2-D ``X`` (with strictly positive ``sigma_X``).
        instance_noise: Optional blur configuration applied to discriminator
            inputs before normalisation.
        generator_step: Current outer generator step for the blur schedule lookup.
        batch_role: Whether this is a ``"real"`` or ``"fake"`` discriminator batch.

    Returns:
        ``(batch, root_indices)``: a PyG ``Batch`` with node features of shape
        ``(sum_r |B_k(root_r)|, d_x + 2)`` (columns ``[X_tilde (d_x), Y_tilde,
        root_marker]``), and a long ``(batch_size,)`` tensor indexing each root in
        the concatenated node axis.

    Raises:
        KeyError: If normalisation keys are missing or a root is absent from the cache.
        TypeError, ValueError: For invalid tensor shapes/dtypes/devices.
    """
    required = {"mu_X", "sigma_X", "mu_Y", "sigma_Y"}
    missing = required.difference(norm_stats.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"norm_stats missing required keys: {missing_str}.")

    if not isinstance(X, Tensor) or not isinstance(Y, Tensor):
        raise TypeError("X and Y must be torch.Tensor objects.")
    if X.ndim not in (1, 2):
        raise ValueError("X must have shape (n,) or (n, d_x).")
    if Y.ndim != 1:
        raise ValueError("Y must have shape (n,).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of nodes (X.shape[0] == Y.shape[0]).")
    if X.ndim == 2 and X.shape[1] < 1:
        raise ValueError("X must have at least one covariate column (d_x >= 1).")
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

    mu_Y = float(norm_stats["mu_Y"])
    sigma_Y = float(norm_stats["sigma_Y"])
    if sigma_Y <= 0.0:
        raise ValueError("sigma_Y must be strictly positive.")
    if X.ndim == 1:
        # Scalar-covariate path (d_x = 1): kept bit-identical to the original.
        mu_X = float(norm_stats["mu_X"])
        sigma_X = float(norm_stats["sigma_X"])
        if sigma_X <= 0.0:
            raise ValueError("sigma_X must be strictly positive.")
    else:
        # Vector-covariate path: per-column centre/scale, shape (d_x,).
        d_x = int(X.shape[1])
        mu_X = torch.as_tensor(norm_stats["mu_X"], dtype=X.dtype, device=X.device)
        sigma_X = torch.as_tensor(norm_stats["sigma_X"], dtype=X.dtype, device=X.device)
        if mu_X.ndim != 1 or int(mu_X.shape[0]) != d_x:
            raise ValueError(f"mu_X must have length d_x={d_x} for a {d_x}-column X.")
        if sigma_X.ndim != 1 or int(sigma_X.shape[0]) != d_x:
            raise ValueError(f"sigma_X must have length d_x={d_x} for a {d_x}-column X.")
        if bool((sigma_X <= 0.0).any()):
            raise ValueError("every sigma_X column must be strictly positive.")

    tau_y = instance_noise_taus(
        instance_noise=instance_noise,
        generator_step=generator_step,
    )
    # IZ Sec 4.2 convolves BOTH observed and simulated ego laws with the SAME outcome
    # noise: the blur is always symmetric (applied identically to real and fake). An
    # asymmetric (real-only) blur targets a criterion no theorem analyses, so it is not
    # offered.
    add_blur = instance_noise is not None and bool(instance_noise.enabled)

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
        if add_blur and sigma_y_raw > 0.0:
            Y_sub = Y_sub + torch.randn_like(Y_sub) * sigma_y_raw

        Y_tilde = (Y_sub - mu_Y) / sigma_Y
        if X.ndim == 1:
            # Scalar-covariate path (d_x = 1): kept bit-identical to the original.
            X_tilde = (X_sub - mu_X) / sigma_X
            root_marker = torch.zeros_like(X_tilde)
            root_marker[root_pos] = 1.0
            features = torch.stack((X_tilde, Y_tilde, root_marker), dim=1)
        else:
            # Vector-covariate path: per-column normalisation, then concatenate the
            # d_x covariate columns with the outcome and root-marker columns.
            X_tilde = (X_sub - mu_X) / sigma_X
            root_marker = torch.zeros_like(Y_tilde)
            root_marker[root_pos] = 1.0
            features = torch.cat(
                (X_tilde, Y_tilde.unsqueeze(1), root_marker.unsqueeze(1)), dim=1
            )

        data_list.append(Data(x=features, edge_index=edge_local))
        root_positions.append(int(root_pos))

    batch = Batch.from_data_list(data_list)
    ptr = batch.ptr[:-1]
    root_offsets = torch.tensor(root_positions, dtype=torch.long, device=ptr.device)
    root_indices = ptr + root_offsets
    return batch, root_indices
