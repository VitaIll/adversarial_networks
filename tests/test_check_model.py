"""Tests for check_model admissibility soundness.

The built-in games must pass on a normal graph; a deliberately *non-contractive*
model (a star public-goods map with a raw-neighbour-SUM aggregate) must FAIL the
operator-inf-norm contraction check — the case a median/random-direction ratio
would wrongly green-light.
"""

from __future__ import annotations

import networkx as nx
import torch

from adversarial_networks.ego import EgoSubstrate
from adversarial_networks.generators import (
    EffortGameGenerator,
    LinearInMeansGenerator,
    NetworkGameGenerator,
    check_model,
)
from adversarial_networks.transforms import Positive


def _substrate(graph: nx.Graph, seed: int = 0) -> EgoSubstrate:
    torch.manual_seed(seed)
    X = torch.randn(graph.number_of_nodes())
    return EgoSubstrate.from_networkx(graph, X, k=2, root_sampler_mode="uniform", seed=seed)


def test_linear_in_means_passes() -> None:
    sub = _substrate(nx.barabasi_albert_graph(80, 3, seed=1))
    report = check_model(LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5), sub, n_probe=80)
    assert bool(report)
    # contraction modulus equals |beta| for the linear-in-means map
    assert abs(report["contraction_modulus"].value - 0.4) < 1e-3
    assert report["shock_monotone_U4"].passed
    assert report["locality_A2"].passed


def test_effort_game_passes() -> None:
    sub = _substrate(nx.barabasi_albert_graph(80, 3, seed=2))
    report = check_model(
        EffortGameGenerator(fix_r=1.0, fix_sigma_sq=None, init_gamma=1.2, init_lambda=0.7, init_mu=0.3),
        sub, n_probe=80,
    )
    assert bool(report)
    assert report["gradients"].passed


class _NonContractive(NetworkGameGenerator):
    """A linear mean-aggregate map ``B = beta*(W*Y) + eps`` with ``beta > 1``.

    Its Jacobian ``A = beta*W`` has operator inf-norm ``||A||_inf = beta*||W||_inf = beta``
    (``W`` is row-stochastic), so ``beta = 1.5`` is non-contractive — the contraction
    check must report ~1.5 and FAIL, even though a median row-ratio looks fine.
    """

    sigma_sq = Positive()

    def __init__(self, beta: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._beta = float(beta)

    def best_response(self, peer_agg, X, shocks):
        return self._beta * peer_agg + shocks


def test_non_contractive_linear_fails_contraction() -> None:
    sub = _substrate(nx.barabasi_albert_graph(60, 2, seed=3))
    model = _NonContractive(beta=1.5, picard_tol=1e-6, picard_max=60)
    report = check_model(model, sub, n_probe=sub.num_nodes)
    assert not bool(report)
    contraction = report["contraction_modulus"]
    assert not contraction.passed
    assert abs(contraction.value - 1.5) < 1e-2  # operator inf-norm equals beta


def test_check_model_is_deterministic() -> None:
    sub = _substrate(nx.barabasi_albert_graph(60, 2, seed=5))
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.3, init_gamma=1.0)
    r1 = check_model(model, sub, n_probe=60, seed=11)
    r2 = check_model(model, sub, n_probe=60, seed=11)
    assert r1["contraction_modulus"].value == r2["contraction_modulus"].value
    assert r1["equilibrium_residual"].value == r2["equilibrium_residual"].value
