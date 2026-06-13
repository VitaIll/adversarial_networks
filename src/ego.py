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

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph, to_undirected

from .root_sampling import RootSampler, build_adjacency_from_edge_index, sample_roots_tensor
from .utils import EgoCache, EgoCacheEntry, build_row_stochastic_W, extract_ego_batch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx
    from torch_geometric.data import Batch

    from .root_sampling import RootSamplingResult
    from .utils import InstanceNoiseConfigLike


class EgoSubstrate:
    """Immutable graph + covariate substrate with rooted-ego batch construction.

    Attributes:
        num_nodes: Number of nodes ``n`` in the (sanitised) graph.
        edge_index: Undirected ``(2, num_edges)`` long edge index on ``device``.
        W: Coalesced sparse-COO row-stochastic ``(n, n)`` float32 matrix.
        X: Dense ``(n,)`` float covariate vector.
        ego_cache: Mapping ``root -> (subset, sub_edge_index, root_pos)`` for every
            node, with ``k``-hop induced subgraphs.
        root_sampler: Configured :class:`~src.root_sampling.RootSampler`.
        k: Ego radius used to build the cache.
        mu_X: Mean of ``X`` (covariate normalisation centre).
        sigma_X: Population std of ``X`` (covariate normalisation scale, > 0).
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
        mu_X: float | None = None,
        sigma_X: float | None = None,
    ) -> None:
        if not isinstance(X, Tensor) or X.ndim != 1 or not torch.is_floating_point(X):
            raise TypeError("X must be a 1-D floating-point tensor of shape (n,).")
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
        self.device = X.device

        if sigma_X is None:
            sigma_X = float(X.std(unbiased=False).item())
        if mu_X is None:
            mu_X = float(X.mean().item())
        if sigma_X <= 1e-12:
            raise ValueError(
                f"sigma_X must be strictly positive (X is near-constant), got {sigma_X}."
            )
        self.mu_X = float(mu_X)
        self.sigma_X = float(sigma_X)
        self.sanitization_report: dict[str, int] = {}

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
            X: ``(n,)`` float covariate vector defining the node count and device.
            k: Ego radius for the cache (must match the discriminator depth).
            root_sampler: Pre-constructed sampler whose ``num_nodes`` equals ``n``.
            ensure_undirected: Symmetrise ``edge_index`` before building ``W``.

        Returns:
            A validated :class:`EgoSubstrate`.

        Raises:
            TypeError, ValueError: On any precondition violation.
        """
        if not isinstance(X, Tensor) or X.ndim != 1 or not torch.is_floating_point(X):
            raise TypeError("X must be a 1-D floating-point tensor of shape (n,).")
        num_nodes = int(X.shape[0])
        device = X.device

        if not isinstance(edge_index, Tensor) or edge_index.dtype != torch.long:
            raise TypeError("edge_index must be a torch.long tensor.")
        edge_index = edge_index.to(device)
        if ensure_undirected:
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = edge_index.contiguous()

        W = build_row_stochastic_W(edge_index=edge_index, num_nodes=num_nodes)
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
        graph: "nx.Graph",
        X: Tensor,
        *,
        k: int,
        root_sampler_mode: str = "uniform",
        exclusion_r: int = 0,
        disjoint_restarts_k: int = 1,
        disjoint_min_batch: int | None = None,
        disjoint_relax_sequence: tuple[int, ...] = (0,),
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
            X: ``(N,)`` float covariates aligned to ``sorted(graph.nodes())``.
            k: Ego radius.
            root_sampler_mode: One of the :class:`~src.root_sampling.RootSampler`
                modes.
            exclusion_r, disjoint_restarts_k, disjoint_min_batch,
            disjoint_relax_sequence, disjoint_fallback: Disjoint-sampler controls
                (ignored for ``"uniform"``).
            seed: Seed for the sampler RNG.

        Returns:
            A validated :class:`EgoSubstrate`.

        Raises:
            ImportError: If NetworkX is unavailable.
            ValueError: If ``X`` length does not match the graph node count, or the
                sanitised graph has no edges.
        """
        import networkx as nx

        if not isinstance(X, Tensor) or X.ndim != 1 or not torch.is_floating_point(X):
            raise TypeError("X must be a 1-D floating-point tensor.")
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

        adjacency = build_adjacency_from_edge_index(edge_index=edge_index.cpu(), num_nodes=num_nodes)
        import numpy as np

        sampler = RootSampler(
            num_nodes=num_nodes,
            mode=root_sampler_mode,  # type: ignore[arg-type]
            exclusion_r=exclusion_r,
            disjoint_restarts_k=disjoint_restarts_k,
            disjoint_min_batch=disjoint_min_batch,
            disjoint_relax_sequence=disjoint_relax_sequence,
            disjoint_fallback=disjoint_fallback,  # type: ignore[arg-type]
            rng=np.random.default_rng(seed),
            adjacency=adjacency if root_sampler_mode != "uniform" else None,
        )
        substrate = cls.from_edge_index(
            edge_index=edge_index, X=X_kept, k=k, root_sampler=sampler, ensure_undirected=False
        )
        substrate.sanitization_report = {
            "self_loops_removed": int(n_selfloops),
            "nodes_dropped": int(n_before - num_nodes),
            "num_nodes": int(num_nodes),
        }
        return substrate

    @staticmethod
    def _build_ego_cache(*, edge_index: Tensor, num_nodes: int, k: int, device: torch.device) -> dict[int, EgoCacheEntry]:
        """Precompute the rooted ``k``-ego cache for every node.

        Each entry is ``(subset, sub_edge_index, root_pos)`` with relabelled local
        node ids, matching the format consumed by
        :func:`~src.utils.extract_ego_batch`.
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
    def sample_roots(self, batch_size: int) -> tuple[Tensor, "RootSamplingResult"]:
        """Sample a batch of root node ids on this substrate's device.

        Args:
            batch_size: Requested number of roots.

        Returns:
            ``(roots_tensor, result)`` where ``roots_tensor`` is a ``(achieved,)``
            long tensor and ``result`` carries sampler diagnostics.
        """
        return sample_roots_tensor(sampler=self.root_sampler, batch_size=batch_size, device=self.device)

    def make_norm_stats(self, Y: Tensor) -> dict[str, float]:
        """Compute the frozen normalisation stats ``{mu_X, sigma_X, mu_Y, sigma_Y}``.

        The covariate stats are the substrate's fixed values; the outcome stats are
        computed from the supplied (observed) outcome vector. These same stats
        normalise both real and simulated batches.

        Args:
            Y: ``(n,)`` float outcome vector (typically the observed ``Y_obs``).

        Returns:
            Normalisation dictionary with strictly positive scales.

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
        instance_noise: "InstanceNoiseConfigLike | None" = None,
    ) -> tuple["Batch", Tensor]:
        """Construct a rooted-ego PyG batch for the given roots and outcomes.

        Thin, validated wrapper over :func:`~src.utils.extract_ego_batch` binding
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
            :func:`~src.utils.extract_ego_batch`.
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
