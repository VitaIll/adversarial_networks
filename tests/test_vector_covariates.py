"""Tests for the multivariate node-covariate generalisation (``X`` of shape ``(n, d_x)``).

Illichmann & Zacchia (2026) Examples 2-3 use a vector covariate effect ``X'_{n,i} gamma``.
These tests exercise the *framework* generalisation around scalar built-ins:

* an end-to-end ``AdversarialEstimator.fit`` on a custom ``NetworkGameGenerator`` with a
  ``(d_x,)`` covariate-effect ``gamma`` (``d_x = 3``), asserting a finite ``status="ok"``
  fit and that ``get_params`` expands the vector ``gamma`` into ``d_x`` indexed keys;
* per-column standardisation of the built ego features (each ``X_tilde`` column ~ N(0,1));
* the scalar (``d_x = 1``) path is bit-identical (a built ego batch is ``(., 3)`` and
  column 0 equals ``(X_sub - mu) / sigma``);
* base ``get_params`` does not truncate a vector parameter.

The graph + batch idiom (``k_hop_subgraph`` + PyG ``Data``/``Batch``) mirrors
``tests/test_core_ego_features.py`` and ``tests/test_discriminator.py``.
"""

from __future__ import annotations

import math
import warnings

import networkx as nx
import torch
from torch import nn
from torch_geometric.utils import from_networkx, k_hop_subgraph, to_undirected

from adversarial_networks.core.ego_features import extract_ego_batch
from adversarial_networks.data import NetworkData
from adversarial_networks.discriminator import RootedMPNNDiscriminator
from adversarial_networks.estimator import AdversarialEstimator
from adversarial_networks.estimator_config import EstimatorConfig
from adversarial_networks.generators import NetworkGameGenerator
from adversarial_networks.transforms import Interval, Positive

D_X = 3


class VectorLinearInMeans(NetworkGameGenerator):
    """Linear-in-means with a *vector* covariate effect ``X @ gamma``, ``gamma`` of shape ``(d_x,)``.

    ``best_response = beta * peer_agg + X @ gamma + shocks`` with ``beta`` capped to
    ``(-0.9, 0.9)`` (an :class:`Interval` transform, so contraction holds for every
    optimiser step) and a Gaussian shock channel (``sigma_sq`` via :class:`Positive`).
    ``gamma`` is a declared ``(d_x,)`` ``nn.Parameter`` (no constraint), so the default
    ``sample_shocks`` (per-node ``(n,)`` Gaussian) and the base ``get_params`` (vector
    expansion) are exercised unchanged.
    """

    beta = Interval(-0.9, 0.9)
    sigma_sq = Positive()

    def __init__(self, d_x: int = D_X, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.gamma = nn.Parameter(torch.zeros(d_x, dtype=torch.float32))

    def constrained_params(self) -> dict[str, torch.Tensor]:
        params = super().constrained_params()  # beta, sigma_sq from the declared transforms
        params["gamma"] = self.gamma
        return params

    def best_response(self, peer_agg: torch.Tensor, X: torch.Tensor, shocks: torch.Tensor) -> torch.Tensor:
        p = self.params()
        return p["beta"] * peer_agg + (X @ self.gamma) + shocks


def _ba_edge_index(n: int, seed: int = 0) -> torch.Tensor:
    graph = nx.barabasi_albert_graph(n, 2, seed=seed)
    return to_undirected(from_networkx(graph).edge_index, num_nodes=n).contiguous()


def _complete_edge_index(n: int) -> torch.Tensor:
    graph = nx.complete_graph(n)
    return to_undirected(from_networkx(graph).edge_index, num_nodes=n).contiguous()


def _set_constrained(model: VectorLinearInMeans, *, beta: float, gamma: list[float]) -> None:
    """Initialise a model to a desired constrained ``(beta, gamma)`` for ground-truth simulation."""
    with torch.no_grad():
        model._raw_beta.copy_(Interval(-0.9, 0.9).inverse(beta))
        model.gamma.copy_(torch.tensor(gamma, dtype=torch.float32))


# --------------------------------------------------------------- end-to-end vector fit
def test_vector_covariate_fit_runs_end_to_end_and_expands_gamma() -> None:
    """A d_x=3 vector-covariate game fits end-to-end with finite params; gamma expands to 3 keys."""
    torch.manual_seed(0)
    n = 80
    edge_index = _ba_edge_index(n, seed=0)
    X = torch.randn(n, D_X, dtype=torch.float32)

    # Ground-truth outcome from a known (beta, gamma).
    truth = VectorLinearInMeans(D_X, picard_tol=1e-7, picard_max=200)
    _set_constrained(truth, beta=0.4, gamma=[1.5, -0.8, 0.4])
    W = NetworkData.from_edge_index(edge_index, X, torch.zeros(n), k=2).topology.W
    torch.manual_seed(1)
    with torch.no_grad():
        y = truth(W, X).to(torch.float32)
    assert bool(torch.isfinite(y).all())

    data = NetworkData.from_edge_index(edge_index, X, y, k=2)
    assert data.topology.d_x == D_X
    assert isinstance(data.topology.mu_X, torch.Tensor) and tuple(data.topology.mu_X.shape) == (D_X,)

    model = VectorLinearInMeans(D_X, picard_tol=1e-7, picard_max=200)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, in_dim=D_X + 2, logit_clip=10.0)
    config = EstimatorConfig(
        max_steps=6, min_steps=0, batch_size=8, n_disc=1, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=100, stability_window=30, seed=7,
    )
    est = AdversarialEstimator(model, disc, config=config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short fit will not converge -> ConvergenceWarning
        est.fit(data)

    assert est.result_.status == "ok"
    assert est.n_iter_ == 6
    # gamma is a (d_x,) vector -> expanded into indexed keys, NOT truncated to one.
    expected_keys = {"beta", "sigma_sq", "gamma[0]", "gamma[1]", "gamma[2]"}
    assert set(est.params_.keys()) == expected_keys
    assert all(math.isfinite(v) for v in est.params_.values())
    # The learned model itself exposes the same expanded surface.
    assert set(est.model_.get_params().keys()) == expected_keys


def test_in_dim_guard_rejects_mismatched_discriminator() -> None:
    """A d_x=3 substrate with a default (in_dim=3) discriminator must fail loudly in fit()."""
    torch.manual_seed(0)
    n = 40
    edge_index = _ba_edge_index(n, seed=1)
    X = torch.randn(n, D_X, dtype=torch.float32)
    y = torch.randn(n, dtype=torch.float32)
    data = NetworkData.from_edge_index(edge_index, X, y, k=2)

    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2)  # in_dim=3, but needs d_x+2=5
    est = AdversarialEstimator(
        VectorLinearInMeans(D_X), disc,
        config=EstimatorConfig(
            max_steps=2, min_steps=0, batch_size=4, n_disc=1, lr_d=1e-3, lr_g=5e-3,
            grad_clip_norm=10.0, convergence_window=10, stability_window=5, seed=1,
        ),
    )
    try:
        est.fit(data)
    except ValueError as exc:
        assert "in_dim" in str(exc) and "d_x + 2" in str(exc)
    else:  # pragma: no cover - guard against a silent contract regression
        raise AssertionError("a d_x mismatch between data and discriminator must raise.")


# ------------------------------------------------------------- per-column standardisation
def test_ego_features_are_standardised_per_column() -> None:
    """On a complete graph (1-hop ego = all nodes), each X_tilde column is ~ N(0, 1).

    A complete graph with ``k = 1`` makes every root's induced ego subgraph the full
    node set, so the per-column-normalised covariate block of the ego batch is exactly
    the population-standardised ``X``: each of the ``d_x`` columns has mean ~ 0 and
    population std ~ 1.
    """
    torch.manual_seed(0)
    n = 50
    edge_index = _complete_edge_index(n)
    X = torch.randn(n, D_X, dtype=torch.float32) * torch.tensor([3.0, 0.5, 10.0]) + 7.0
    y = torch.randn(n, dtype=torch.float32)
    data = NetworkData.from_edge_index(edge_index, X, y, k=1)

    roots, _ = data.topology.sample_roots(1)
    norm = data.topology.make_norm_stats(y)
    batch, _ = data.topology.build_batch(roots, y, norm, step=0, role="real")

    assert batch.x.shape[1] == D_X + 2  # [X_tilde (d_x), Y_tilde, root_marker]
    x_tilde = batch.x[:, :D_X]
    assert x_tilde.shape[0] == n  # the single ego covers all nodes on a complete graph
    assert torch.allclose(x_tilde.mean(dim=0), torch.zeros(D_X), atol=1e-4)
    assert torch.allclose(x_tilde.std(dim=0, unbiased=False), torch.ones(D_X), atol=1e-4)


# -------------------------------------------------------- scalar path bit-identity (H)
def test_scalar_path_is_unchanged() -> None:
    """A 1-D X yields a (num_nodes, 3) feature matrix; column 0 == (X_sub - mu) / sigma.

    This is the bit-identity guarantee for the scalar (``d_x = 1``) case: the feature
    layout is exactly ``[X_tilde, Y_tilde, root_marker]`` with three columns, and the
    covariate column is the scalar-standardised ``X`` restricted to the ego subset.
    """
    torch.manual_seed(11)
    n = 14
    graph = nx.path_graph(n)
    edge_index = to_undirected(from_networkx(graph).edge_index, num_nodes=n)

    ego_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for root in range(n):
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=root, num_hops=2, edge_index=edge_index, relabel_nodes=True, num_nodes=n,
        )
        ego_cache[root] = (subset, sub_edge_index, int(mapping.item()))

    roots = torch.tensor([0, 3, 7, 10], dtype=torch.long)
    X = torch.randn(n, dtype=torch.float32)
    Y = torch.randn(n, dtype=torch.float32)
    mu_X = float(X.mean().item())
    sigma_X = float(X.std(unbiased=False).item())
    norm_stats = {
        "mu_X": mu_X, "sigma_X": sigma_X,
        "mu_Y": float(Y.mean().item()), "sigma_Y": float(Y.std(unbiased=False).item()),
    }

    batch, _ = extract_ego_batch(roots=roots, ego_cache=ego_cache, X=X, Y=Y, norm_stats=norm_stats)

    # Exactly three feature columns for a scalar covariate.
    assert batch.x.ndim == 2 and batch.x.shape[1] == 3

    # Column 0 is the scalar-standardised X over the concatenated ego subsets.
    expected_cols: list[torch.Tensor] = []
    for root in [int(r) for r in roots.tolist()]:
        subset, _, _ = ego_cache[root]
        expected_cols.append((X.index_select(0, subset) - mu_X) / sigma_X)
    expected_x_tilde = torch.cat(expected_cols, dim=0)
    assert torch.equal(batch.x[:, 0], expected_x_tilde)


# --------------------------------------------------------- base get_params no-truncation (I)
def test_base_get_params_expands_vector_parameter_without_truncation() -> None:
    """A vector constrained parameter is expanded into indexed keys, never truncated."""
    class VecParamGame(NetworkGameGenerator):
        sigma_sq = Positive()

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.gamma = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))

        def constrained_params(self) -> dict[str, torch.Tensor]:
            params = super().constrained_params()  # sigma_sq
            params["gamma"] = self.gamma
            return params

        def best_response(self, peer_agg, X, shocks):  # noqa: ANN001 - test stub
            return peer_agg + shocks

    g = VecParamGame()
    params = g.get_params()
    # The scalar sigma_sq stays a single key; the (4,) gamma expands to four indexed keys.
    assert set(params) == {"sigma_sq", "gamma[0]", "gamma[1]", "gamma[2]", "gamma[3]"}
    assert "gamma" not in params  # no un-indexed (truncated) scalar key
    assert params["gamma[0]"] == 1.0
    assert params["gamma[1]"] == 2.0
    assert params["gamma[2]"] == 3.0
    assert params["gamma[3]"] == 4.0
    assert abs(params["sigma_sq"] - 1.0) < 1e-6


def test_scalar_get_params_keeps_single_key() -> None:
    """A scalar (numel==1) parameter still maps to a single un-indexed key (no regression)."""
    class ScalarParamGame(NetworkGameGenerator):
        sigma_sq = Positive()

        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.gamma = nn.Parameter(torch.tensor([2.5], dtype=torch.float32))  # numel == 1

        def constrained_params(self) -> dict[str, torch.Tensor]:
            params = super().constrained_params()
            params["gamma"] = self.gamma
            return params

        def best_response(self, peer_agg, X, shocks):  # noqa: ANN001 - test stub
            return peer_agg + shocks

    params = ScalarParamGame().get_params()
    assert set(params) == {"sigma_sq", "gamma"}  # a 1-element vector is a scalar key
    assert abs(params["gamma"] - 2.5) < 1e-6
