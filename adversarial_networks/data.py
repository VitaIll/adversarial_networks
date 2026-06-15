"""The observed-network data container (the domain object).

:class:`NetworkData` bundles an observed graph, node covariates ``X``, and the
**mandatory** equilibrium outcome ``y`` — the network analogue of ``DoubleMLData``
(where the outcome column is required). It is a thin owner of a *private*
:class:`~adversarial_networks.ego.EgoSubstrate` (topology + ``W`` + ``X`` + the
precomputed ``k``-ego cache + the root sampler); the substrate is reusable across
Monte Carlo realisations, but a ``NetworkData`` always carries exactly one outcome.

Construction validates at the boundary and **validates before assigning** (no
half-built object on a validation error — the DoubleML #144 anti-pattern): ``X`` is a
finite ``float32`` tensor of shape ``(n,)`` (scalar covariate) or ``(n, d_x)`` (vector
covariate) and ``y`` a finite 1-D ``float32`` tensor, both of the node count and on
one device. Graph sanitisation (self-loops removed, restricted to the largest
connected component, relabelled) happens inside the substrate, and ``y`` is re-indexed
onto the kept nodes identically to ``X`` (whose rows are re-indexed in the substrate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .ego import EgoSubstrate
from .sampling import RootSampler

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

    from .contracts import StructuralModel


def _validate_vector(name: str, tensor: Tensor, *, length: int | None = None,
                     device: torch.device | None = None) -> None:
    """Reject anything but a finite 1-D ``float32`` vector (no silent coercion)."""
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1-D (shape (n,)), got shape {tuple(tensor.shape)}.")
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"{name} must be float32 (the estimation contract rejects rather than "
            f"silently downcasts), got {tensor.dtype}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values (NaN/inf).")
    if length is not None and int(tensor.shape[0]) != length:
        raise ValueError(f"{name} must have length {length}, got {int(tensor.shape[0])}.")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}.")


def _validate_covariates(name: str, tensor: Tensor, *, length: int | None = None,
                         device: torch.device | None = None) -> None:
    """Reject anything but a finite ``float32`` covariate tensor (no silent coercion).

    Generalises :func:`_validate_vector` to the covariate ``X``, which may be a single
    scalar per node (shape ``(n,)``, ``d_x = 1``) or a vector per node (shape
    ``(n, d_x)``, ``d_x >= 1``). ``length`` constrains the node count (row axis).
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim not in (1, 2):
        raise ValueError(
            f"{name} must have shape (n,) or (n, d_x), got shape {tuple(tensor.shape)}."
        )
    if tensor.ndim == 2 and int(tensor.shape[1]) < 1:
        raise ValueError(f"{name} must have at least one covariate column (d_x >= 1).")
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"{name} must be float32 (the estimation contract rejects rather than "
            f"silently downcasts), got {tensor.dtype}."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values (NaN/inf).")
    if length is not None and int(tensor.shape[0]) != length:
        raise ValueError(f"{name} must have length {length}, got {int(tensor.shape[0])}.")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}.")


class NetworkData:
    """Observed graph + covariates + (mandatory) equilibrium outcome.

    Build with :meth:`from_networkx` (messy graph, sanitised), :meth:`from_edge_index`
    (clean contiguous graph), or :meth:`simulate` (simulate the outcome from any
    model on a topology). The estimator consumes a ``NetworkData`` via ``fit(data)``.
    """

    def __init__(self, topology: EgoSubstrate, y: Tensor) -> None:
        if not isinstance(topology, EgoSubstrate):
            raise TypeError("topology must be an EgoSubstrate instance.")
        _validate_vector("y", y, length=topology.num_nodes, device=topology.device)
        # validate-before-assign: both checks pass before any attribute is set.
        self._topology = topology
        self._y = y.detach()

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        X: Tensor,
        y: Tensor,
        *,
        k: int,
        root_sampler_mode: str = "uniform",
        exclusion_r: int | None = None,
        disjoint_restarts_k: int = 1,
        disjoint_min_batch: int | None = None,
        disjoint_relax_sequence: tuple[int, ...] | None = None,
        disjoint_fallback: str = "best",
        seed: int = 0,
    ) -> NetworkData:
        """Build from a (possibly messy) NetworkX graph; ``X``/``y`` align to ``sorted(nodes)``.

        The graph is sanitised inside the substrate (self-loops removed, restricted
        to the largest connected component, relabelled); ``X`` *and* ``y`` are
        re-indexed onto the kept nodes. The sanitisation counts are on
        :attr:`sanitization_report`.

        In a disjoint ``root_sampler_mode`` an unset ``exclusion_r`` /
        ``disjoint_relax_sequence`` (``None``) defaults to the vertex-disjoint radius
        ``2 * k``, so the sampled radius-``k`` egos are near-independent (fn. 26);
        explicit radii below ``2 * k`` trade independence for batch fill and emit a
        ``RuntimeWarning`` from the substrate. Both controls are inherited from
        :meth:`EgoSubstrate.from_networkx`.
        """
        n_graph = graph.number_of_nodes()
        _validate_covariates("X", X, length=n_graph)
        _validate_vector("y", y, length=n_graph, device=X.device)
        topology = EgoSubstrate.from_networkx(
            graph, X, k=k, root_sampler_mode=root_sampler_mode, exclusion_r=exclusion_r,
            disjoint_restarts_k=disjoint_restarts_k, disjoint_min_batch=disjoint_min_batch,
            disjoint_relax_sequence=disjoint_relax_sequence, disjoint_fallback=disjoint_fallback,
            seed=seed,
        )
        if topology.kept_positions is not None:
            y_kept = y.index_select(0, topology.kept_positions.to(y.device)).contiguous()
        else:  # pragma: no cover - from_networkx always sets kept_positions
            y_kept = y
        return cls(topology, y_kept)

    @classmethod
    def from_edge_index(
        cls,
        edge_index: Tensor,
        X: Tensor,
        y: Tensor,
        *,
        k: int,
        root_sampler: RootSampler | None = None,
        ensure_undirected: bool = True,
    ) -> NetworkData:
        """Build from a clean, contiguously labelled edge index (no sanitisation)."""
        n = int(X.shape[0])
        _validate_covariates("X", X)
        _validate_vector("y", y, length=n, device=X.device)
        if root_sampler is None:
            import numpy as np

            root_sampler = RootSampler(num_nodes=n, mode="uniform", rng=np.random.default_rng(0))
        topology = EgoSubstrate.from_edge_index(
            edge_index, X, k=k, root_sampler=root_sampler, ensure_undirected=ensure_undirected
        )
        return cls(topology, y)

    @classmethod
    def simulate(
        cls,
        graph_or_edge_index,
        X: Tensor,
        model: StructuralModel,
        *,
        k: int,
        seed: int | None = None,
        **sampler_kw,
    ) -> NetworkData:
        """Build a topology, simulate the outcome from ``model``, and wrap.

        The general "simulate on my network" path: works with *any* model satisfying
        the :class:`~adversarial_networks.contracts.StructuralModel` protocol.
        """
        import networkx as nx

        if isinstance(graph_or_edge_index, Tensor):
            n = int(X.shape[0])
            import numpy as np

            sampler = RootSampler(num_nodes=n, mode="uniform", rng=np.random.default_rng(seed or 0))
            topology = EgoSubstrate.from_edge_index(graph_or_edge_index, X, k=k, root_sampler=sampler)
        elif isinstance(graph_or_edge_index, nx.Graph):
            topology = EgoSubstrate.from_networkx(graph_or_edge_index, X, k=k, seed=seed or 0, **sampler_kw)
        else:
            raise TypeError("graph_or_edge_index must be a networkx.Graph or an edge-index Tensor.")

        device = topology.device
        model = model.to(device)  # type: ignore[attr-defined]
        if seed is not None:
            torch.manual_seed(int(seed))
        with torch.no_grad():
            y_sim = model(topology.W, topology.X).detach()
        return cls(topology, y_sim.to(torch.float32))

    # ------------------------------------------------------------------ accessors
    @property
    def num_nodes(self) -> int:
        return self._topology.num_nodes

    @property
    def X(self) -> Tensor:
        return self._topology.X

    @property
    def y(self) -> Tensor:
        return self._y

    @property
    def k(self) -> int:
        return self._topology.k

    @property
    def device(self) -> torch.device:
        return self._topology.device

    @property
    def topology(self) -> EgoSubstrate:
        """The private :class:`EgoSubstrate` (advanced: EDA / plotting needs ``W``, ``edge_index``)."""
        return self._topology

    @property
    def W(self) -> Tensor:
        """The row-stochastic interaction matrix (convenience for ``check_model`` etc.)."""
        return self._topology.W

    @property
    def sanitization_report(self) -> dict[str, int]:
        return dict(self._topology.sanitization_report)

    def to_networkx(self) -> nx.Graph:
        """Rebuild a NetworkX graph from the (sanitised) edge index, for graph plots."""
        import networkx as nx

        edges = self._topology.edge_index.t().cpu().numpy()
        graph = nx.Graph()
        graph.add_nodes_from(range(self._topology.num_nodes))
        graph.add_edges_from((int(u), int(v)) for u, v in edges)
        return graph

    def __repr__(self) -> str:
        return f"NetworkData(num_nodes={self.num_nodes}, k={self.k}, device={self.device})"
