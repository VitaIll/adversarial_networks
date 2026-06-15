"""The ego-object data substrate for adversarial structural estimation.

The :class:`EgoSubstrate` bundles the *immutable* estimation substrate — graph
topology, the row-stochastic interaction matrix ``W``, covariates ``X``, the
precomputed rooted ``k``-ego cache, the root sampler, and the covariate
normalisation statistics — together with the rooted-ego batch construction. It is
the object form of the paper's "efficient focal-neighbourhood data construction"
(Section 4.1): build it once, reuse it across many estimation runs.

Design:
    * The substrate is *topological + covariate* only. It is deliberately free of
      any outcome vector ``Y`` so that one substrate can be shared across Monte
      Carlo realisations that each draw a different observed outcome. Outcome
      normalisation (``mu_Y``/``sigma_Y``) is computed per realisation via
      :meth:`make_norm_stats` and supplied to :meth:`build_batch`.
    * Construction validates internal consistency (sizes, devices, cache
      coverage) and fails loudly with an attributable message, rather than
      surfacing a cryptic error deep inside a training step.

References:
    Illichmann & Zacchia (2026), Section 4.1 (computational primitives) and
    Section 2.1 (ego objects).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph, to_undirected

from .core.ego_features import EgoCache, EgoCacheEntry, extract_ego_batch
from .core.graph import adjacency_lists_from_edge_index, row_stochastic_weights
from .sampling import RootSampler, sample_roots_tensor


def _sampler_effective_exclusion_r(root_sampler: RootSampler) -> int:
    """Strictest exclusion radius a disjoint sampler enforces between roots.

    For the single-radius modes this is ``exclusion_r``; for ``disjoint_relax`` it
    is the *largest* radius the relax ladder ever attempts (the best-case packing,
    hence the best-case ego independence the configuration can deliver). The caller
    is expected to have already checked ``root_sampler.mode != "uniform"``.
    """
    if root_sampler.mode == "disjoint_relax":
        return max(root_sampler.disjoint_relax_sequence)
    return root_sampler.exclusion_r


def _warn_if_egos_overlap(root_sampler: RootSampler, k: int) -> None:
    """Warn once if a disjoint sampler's exclusion radius cannot make k-egos disjoint.

    Two radius-``k`` ego balls are vertex-disjoint iff the distance between their
    centres exceeds ``2k``; the packer only guarantees ``dist(u, v) > exclusion_r``.
    So when the effective exclusion radius is ``< 2k`` the sampled radius-``k`` egos
    can share vertices and the near-independence the disjoint modes rely on does not
    hold. ``uniform`` mode does no packing and is exempt.
    """
    if root_sampler.mode == "uniform":
        return
    effective_r = _sampler_effective_exclusion_r(root_sampler)
    required_r = 2 * int(k)
    if effective_r < required_r:
        warnings.warn(
            f"Disjoint root sampler (mode={root_sampler.mode!r}) has effective "
            f"exclusion radius {effective_r} < 2*k = {required_r} (k={int(k)}); the "
            "sampled radius-k egos are NOT vertex-disjoint (they can share vertices), "
            "so the near-independence the disjoint packing relies on does not hold. "
            f"Set exclusion_r >= {required_r} for vertex-disjoint egos.",
            RuntimeWarning,
            stacklevel=2,
        )

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx
    from torch_geometric.data import Batch

    from .core.types import InstanceNoiseConfigLike
    from .sampling import RootSamplingResult


class EgoSubstrate:
    """Immutable graph + covariate substrate with rooted-ego batch construction.

    Attributes:
        num_nodes: Number of nodes ``n`` in the (sanitised) graph.
        edge_index: Undirected ``(2, num_edges)`` long edge index on ``device``.
        W: Coalesced sparse-COO row-stochastic ``(n, n)`` float32 matrix.
        X: Dense ``(n,)`` (scalar covariate) or ``(n, d_x)`` (vector covariate) float
            covariate tensor.
        d_x: Number of covariate columns (``X.shape[1]`` for a 2-D ``X``, else ``1``).
        ego_cache: Mapping ``root -> (subset, sub_edge_index, root_pos)`` for every
            node, with ``k``-hop induced subgraphs.
        root_sampler: Configured :class:`~adversarial_networks.sampling.RootSampler`.
        k: Ego radius used to build the cache.
        mu_X: Mean of ``X`` (covariate normalisation centre): a scalar ``float`` for a
            1-D ``X``, a per-column ``(d_x,)`` tensor for a 2-D ``X``.
        sigma_X: Population std of ``X`` (covariate normalisation scale, > 0): a scalar
            ``float`` for a 1-D ``X``, a per-column ``(d_x,)`` tensor for a 2-D ``X``.
        device: Torch device hosting the tensors.
    """

    def __init__(
        self,
        *,
        edge_index: Tensor,
        X: Tensor,
        W: Tensor,
        ego_cache: EgoCache,
        root_sampler: RootSampler,
        k: int,
        mu_X: float | Tensor | None = None,
        sigma_X: float | Tensor | None = None,
    ) -> None:
        if not isinstance(X, Tensor) or X.ndim not in (1, 2) or not torch.is_floating_point(X):
            raise TypeError("X must be a floating-point tensor of shape (n,) or (n, d_x).")
        if X.ndim == 2 and int(X.shape[1]) < 1:
            raise ValueError("X must have at least one covariate column (d_x >= 1).")
        num_nodes = int(X.shape[0])
        if num_nodes <= 0:
            raise ValueError("X must be non-empty.")
        if not isinstance(edge_index, Tensor) or edge_index.dtype != torch.long:
            raise TypeError("edge_index must be a torch.long tensor.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")
        if not (isinstance(W, Tensor) and W.is_sparse and W.layout == torch.sparse_coo):
            raise TypeError("W must be a sparse COO tensor.")
        if tuple(W.shape) != (num_nodes, num_nodes):
            raise ValueError(f"W must have shape (n, n)=({num_nodes}, {num_nodes}), got {tuple(W.shape)}.")
        if not isinstance(root_sampler, RootSampler):
            raise TypeError("root_sampler must be a RootSampler instance.")
        if root_sampler.num_nodes != num_nodes:
            raise ValueError(
                f"root_sampler.num_nodes ({root_sampler.num_nodes}) must equal n ({num_nodes})."
            )
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        if len(ego_cache) != num_nodes:
            raise ValueError(
                f"ego_cache must cover all {num_nodes} nodes, got {len(ego_cache)} entries."
            )
        if X.device != edge_index.device or X.device != W.device:
            raise ValueError("X, edge_index, and W must be on the same device.")

        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.X = X
        self.W = W
        self.ego_cache = ego_cache
        self.root_sampler = root_sampler
        self.k = int(k)
        # One-time vertex-disjointness check: a disjoint sampler only yields
        # vertex-disjoint radius-k egos when its exclusion radius is >= 2k (fn. 26).
        # This catches a sampler built directly by a user and handed in here.
        _warn_if_egos_overlap(root_sampler, self.k)
        self.device = X.device
        self.d_x = int(X.shape[1]) if X.ndim == 2 else 1

        if X.ndim == 1:
            # Scalar-covariate path (d_x = 1): scalar float stats, kept bit-identical.
            if sigma_X is None:
                sigma_X = float(X.std(unbiased=False).item())
            if mu_X is None:
                mu_X = float(X.mean().item())
            if float(sigma_X) <= 1e-12:
                raise ValueError(
                    f"sigma_X must be strictly positive (X is near-constant), got {sigma_X}."
                )
            self.mu_X: float | Tensor = float(mu_X)
            self.sigma_X: float | Tensor = float(sigma_X)
        else:
            # Vector-covariate path: per-column (d_x,) centre/scale on X's device.
            mu_vec = X.mean(dim=0) if mu_X is None else torch.as_tensor(
                mu_X, dtype=X.dtype, device=X.device
            )
            sigma_vec = X.std(dim=0, unbiased=False) if sigma_X is None else torch.as_tensor(
                sigma_X, dtype=X.dtype, device=X.device
            )
            if mu_vec.ndim != 1 or int(mu_vec.shape[0]) != self.d_x:
                raise ValueError(f"mu_X must have length d_x={self.d_x}.")
            if sigma_vec.ndim != 1 or int(sigma_vec.shape[0]) != self.d_x:
                raise ValueError(f"sigma_X must have length d_x={self.d_x}.")
            near_constant = (sigma_vec <= 1e-12).nonzero(as_tuple=True)[0]
            if int(near_constant.numel()) > 0:
                cols = [int(c) for c in near_constant.tolist()]
                raise ValueError(
                    f"sigma_X must be strictly positive per column (X near-constant in "
                    f"column(s) {cols})."
                )
            self.mu_X = mu_vec.detach()
            self.sigma_X = sigma_vec.detach()
        self.sanitization_report: dict[str, int] = {}
        # Original-node positions kept after sanitisation (set by from_networkx);
        # lets a caller (NetworkData) re-index an outcome vector identically to X.
        self.kept_positions: Tensor | None = None

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_edge_index(
        cls,
        edge_index: Tensor,
        X: Tensor,
        *,
        k: int,
        root_sampler: RootSampler,
        ensure_undirected: bool = True,
    ) -> EgoSubstrate:
        """Build a substrate from a clean, contiguously labelled edge index.

        Preconditions (validated): node ids are ``0..n-1`` with ``n = len(X)``,
        every node has positive degree, and ``edge_index`` lives on ``X.device``.
        Builds ``W`` and the ``k``-ego cache; wraps the provided ``root_sampler``.

        Args:
            edge_index: ``(2, num_edges)`` long edge index. If ``ensure_undirected``
                is true (default) it is symmetrised.
            X: ``(n,)`` (scalar covariate) or ``(n, d_x)`` (vector covariate) float
                covariate tensor defining the node count and device.
            k: Ego radius for the cache (must match the discriminator depth).
            root_sampler: Pre-constructed sampler whose ``num_nodes`` equals ``n``.
            ensure_undirected: Symmetrise ``edge_index`` before building ``W``.

        Returns:
            A validated :class:`EgoSubstrate`.

        Raises:
            TypeError, ValueError: On any precondition violation.
        """
        if not isinstance(X, Tensor) or X.ndim not in (1, 2) or not torch.is_floating_point(X):
            raise TypeError("X must be a floating-point tensor of shape (n,) or (n, d_x).")
        num_nodes = int(X.shape[0])
        device = X.device

        if not isinstance(edge_index, Tensor) or edge_index.dtype != torch.long:
            raise TypeError("edge_index must be a torch.long tensor.")
        edge_index = edge_index.to(device)
        if ensure_undirected:
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = edge_index.contiguous()

        W = row_stochastic_weights(edge_index=edge_index, num_nodes=num_nodes)
        ego_cache = cls._build_ego_cache(edge_index=edge_index, num_nodes=num_nodes, k=k, device=device)
        return cls(
            edge_index=edge_index,
            X=X,
            W=W,
            ego_cache=ego_cache,
            root_sampler=root_sampler,
            k=k,
        )

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        X: Tensor,
        *,
        k: int,
        root_sampler_mode: str = "uniform",
        exclusion_r: int | None = None,
        disjoint_restarts_k: int = 1,
        disjoint_min_batch: int | None = None,
        disjoint_relax_sequence: tuple[int, ...] | None = None,
        disjoint_fallback: str = "best",
        seed: int = 0,
    ) -> EgoSubstrate:
        """Build a substrate from a (possibly messy) NetworkX graph.

        Sanitises the graph (removes self-loops, restricts to the largest
        connected component, relabels to ``0..m-1``) and re-indexes ``X`` onto the
        kept nodes, then constructs the root sampler and the substrate. The number
        of self-loops removed and nodes dropped are recorded on
        :attr:`sanitization_report` for observability.

        Args:
            graph: Input NetworkX graph; nodes are taken in sorted order to align
                with ``X``.
            X: ``(N,)`` or ``(N, d_x)`` float covariates aligned to
                ``sorted(graph.nodes())`` (rows are re-indexed onto the kept nodes).
            k: Ego radius.
            root_sampler_mode: One of the
                :class:`~adversarial_networks.sampling.RootSampler` modes.
            exclusion_r: Exclusion radius for the disjoint modes. When left at its
                default (``None``) in a disjoint mode it derives the *vertex-disjoint*
                radius ``2 * k`` (two radius-``k`` ego balls are disjoint iff their
                centres are more than ``2k`` apart; fn. 26), so the sampled egos are
                near-independent. Ignored for ``"uniform"``. Passing an explicit value
                ``< 2 * k`` in a disjoint mode trades independence for batch fill and
                emits a ``RuntimeWarning`` (the egos can then share vertices).
            disjoint_relax_sequence: Radius ladder for ``disjoint_relax``. When left at
                its default (``None``) in ``disjoint_relax`` mode it derives ``(2 * k,)``
                — the vertex-disjoint radius. Radii below ``2 * k`` trade independence
                for batch fill (and warn, as above).
            disjoint_restarts_k, disjoint_min_batch, disjoint_fallback: Remaining
                disjoint-sampler controls (ignored for ``"uniform"``).
            seed: Seed for the sampler RNG.

        Returns:
            A validated :class:`EgoSubstrate`.

        Raises:
            ImportError: If NetworkX is unavailable.
            ValueError: If ``X`` length does not match the graph node count, or the
                sanitised graph has no edges.
        """
        import networkx as nx

        is_disjoint = root_sampler_mode != "uniform"
        # k-aware defaults: an unset radius in a disjoint mode becomes the
        # vertex-disjoint radius 2*k. Outside disjoint modes the radii are inert,
        # so an unset value is resolved to a harmless concrete RootSampler default.
        resolved_exclusion_r = (
            (2 * int(k) if is_disjoint else 0) if exclusion_r is None else int(exclusion_r)
        )
        resolved_relax_sequence = (
            ((2 * int(k),) if is_disjoint else (0,))
            if disjoint_relax_sequence is None
            else tuple(int(radius) for radius in disjoint_relax_sequence)
        )

        if not isinstance(X, Tensor) or X.ndim not in (1, 2) or not torch.is_floating_point(X):
            raise TypeError("X must be a floating-point tensor of shape (N,) or (N, d_x).")
        nodes_sorted = sorted(graph.nodes())
        if int(X.shape[0]) != len(nodes_sorted):
            raise ValueError(
                f"X length ({int(X.shape[0])}) must match graph node count ({len(nodes_sorted)})."
            )

        old_index = {node: pos for pos, node in enumerate(nodes_sorted)}
        clean = graph.copy()
        n_selfloops = nx.number_of_selfloops(clean)
        if n_selfloops > 0:
            clean.remove_edges_from(nx.selfloop_edges(clean))

        n_before = clean.number_of_nodes()
        if clean.number_of_nodes() > 0 and not nx.is_connected(clean):
            gcc = max(nx.connected_components(clean), key=len)
            clean = clean.subgraph(gcc).copy()
        kept_nodes = sorted(clean.nodes())
        if not kept_nodes:
            raise ValueError("Graph is empty after sanitization.")

        keep_positions = torch.tensor([old_index[node] for node in kept_nodes], dtype=torch.long)
        X_kept = X.index_select(0, keep_positions.to(X.device)).contiguous()

        relabel = {node: pos for pos, node in enumerate(kept_nodes)}
        clean = nx.relabel_nodes(clean, relabel, copy=True)
        num_nodes = clean.number_of_nodes()

        edge_pairs = list(clean.edges())
        if not edge_pairs:
            raise ValueError("Graph has no edges after sanitization.")
        edge_tensor = torch.tensor(edge_pairs, dtype=torch.long, device=X.device).t().contiguous()
        edge_index = to_undirected(edge_tensor, num_nodes=num_nodes).contiguous()

        adjacency = adjacency_lists_from_edge_index(edge_index=edge_index.cpu(), num_nodes=num_nodes)
        import numpy as np

        sampler = RootSampler(
            num_nodes=num_nodes,
            mode=root_sampler_mode,  # type: ignore[arg-type]
            exclusion_r=resolved_exclusion_r,
            disjoint_restarts_k=disjoint_restarts_k,
            disjoint_min_batch=disjoint_min_batch,
            disjoint_relax_sequence=resolved_relax_sequence,
            disjoint_fallback=disjoint_fallback,  # type: ignore[arg-type]
            rng=np.random.default_rng(seed),
            adjacency=adjacency if is_disjoint else None,
        )
        substrate = cls.from_edge_index(
            edge_index=edge_index, X=X_kept, k=k, root_sampler=sampler, ensure_undirected=False
        )
        substrate.sanitization_report = {
            "self_loops_removed": int(n_selfloops),
            "nodes_dropped": int(n_before - num_nodes),
            "num_nodes": int(num_nodes),
        }
        substrate.kept_positions = keep_positions
        return substrate

    @staticmethod
    def _build_ego_cache(*, edge_index: Tensor, num_nodes: int, k: int, device: torch.device) -> dict[int, EgoCacheEntry]:
        """Precompute the rooted ``k``-ego cache for every node.

        Each entry is ``(subset, sub_edge_index, root_pos)`` with relabelled local
        node ids, matching the format consumed by
        :func:`~adversarial_networks.core.ego_features.extract_ego_batch`.
        """
        ego_cache: dict[int, EgoCacheEntry] = {}
        for root in range(num_nodes):
            subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=root,
                num_hops=k,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            ego_cache[root] = (subset.to(device), sub_edge_index.to(device), int(mapping.item()))
        return ego_cache

    # ------------------------------------------------------------------- batching
    def sample_roots(self, batch_size: int) -> tuple[Tensor, RootSamplingResult]:
        """Sample a batch of root node ids on this substrate's device.

        Args:
            batch_size: Requested number of roots.

        Returns:
            ``(roots_tensor, result)`` where ``roots_tensor`` is a ``(achieved,)``
            long tensor and ``result`` carries sampler diagnostics.
        """
        return sample_roots_tensor(sampler=self.root_sampler, batch_size=batch_size, device=self.device)

    def make_norm_stats(self, Y: Tensor) -> dict[str, float | Tensor]:
        """Compute the frozen normalisation stats ``{mu_X, sigma_X, mu_Y, sigma_Y}``.

        The covariate stats are the substrate's fixed values (scalar floats for a 1-D
        ``X``, per-column ``(d_x,)`` tensors for a 2-D ``X``); the outcome stats are
        computed from the supplied (observed) outcome vector. These same stats
        normalise both real and simulated batches.

        Args:
            Y: ``(n,)`` float outcome vector (typically the observed ``Y_obs``).

        Returns:
            Normalisation dictionary with strictly positive scales. ``mu_Y``/``sigma_Y``
            are floats; ``mu_X``/``sigma_X`` mirror the substrate (float or ``(d_x,)``).

        Raises:
            ValueError: If ``Y`` is the wrong shape or near-constant.
        """
        if not isinstance(Y, Tensor) or Y.ndim != 1 or int(Y.shape[0]) != self.num_nodes:
            raise ValueError(f"Y must be a 1-D tensor of length {self.num_nodes}.")
        sigma_Y = float(Y.std(unbiased=False).item())
        if sigma_Y <= 1e-10:
            raise ValueError(f"sigma_Y must be strictly positive (Y near-constant), got {sigma_Y}.")
        return {
            "mu_X": self.mu_X,
            "sigma_X": self.sigma_X,
            "mu_Y": float(Y.mean().item()),
            "sigma_Y": sigma_Y,
        }

    def build_batch(
        self,
        roots: Tensor,
        Y: Tensor,
        norm_stats: dict[str, float],
        *,
        step: int,
        role: str,
        instance_noise: InstanceNoiseConfigLike | None = None,
    ) -> tuple[Batch, Tensor]:
        """Construct a rooted-ego PyG batch for the given roots and outcomes.

        Thin, validated wrapper over
        :func:`~adversarial_networks.core.ego_features.extract_ego_batch` binding
        this substrate's ego cache and covariates.

        Args:
            roots: ``(batch,)`` long tensor of root node ids.
            Y: ``(n,)`` outcome vector to read ego-outcomes from (``Y_obs`` for a
                real batch, ``Y_sim`` for a fake batch).
            norm_stats: Frozen normalisation stats from :meth:`make_norm_stats`.
            step: Outer generator step (for the instance-noise schedule).
            role: ``"real"`` or ``"fake"``.
            instance_noise: Optional blur configuration applied before normalisation.

        Returns:
            ``(batch, root_indices)`` as produced by
            :func:`~adversarial_networks.core.ego_features.extract_ego_batch`.
        """
        return extract_ego_batch(
            roots=roots,
            ego_cache=self.ego_cache,
            X=self.X,
            Y=Y,
            norm_stats=norm_stats,
            instance_noise=instance_noise,
            generator_step=step,
            batch_role=role,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return (
            f"EgoSubstrate(num_nodes={self.num_nodes}, k={self.k}, "
            f"mode={self.root_sampler.mode!r}, device={self.device})"
        )
