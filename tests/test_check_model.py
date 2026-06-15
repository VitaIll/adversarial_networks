"""Tests for check_model admissibility soundness.

The built-in games must pass on a normal graph; a deliberately *non-contractive*
model (a star public-goods map with a raw-neighbour-SUM aggregate) must FAIL the
operator-inf-norm contraction check — the case a median/random-direction ratio
would wrongly green-light.

Also covered: the vector-covariate (``d_x > 1``) generalisation, the interaction-radius
locality generalisation (a genuine 2-hop model fails at ``r0 = 1`` but passes at
``r0 = 2``), the degenerate-probe guard (negligible peer interaction), and the (M)
moment-condition observability helpers (``estimate_branching`` /
``moment_condition_margin``).
"""

from __future__ import annotations

import math

import networkx as nx
import torch
from torch import nn

from adversarial_networks.ego import EgoSubstrate
from adversarial_networks.generators import (
    EffortGameGenerator,
    LinearInMeansGenerator,
    NetworkGameGenerator,
    check_model,
    estimate_branching,
    moment_condition_margin,
)
from adversarial_networks.transforms import Interval, Positive


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


# --------------------------------------------------------------- vector covariate (d_x>1)
class _VectorLinearInMeans(NetworkGameGenerator):
    """Linear-in-means with a *vector* covariate effect ``X @ gamma`` (``gamma`` of shape ``(d_x,)``).

    ``best_response = beta * peer_agg + X @ gamma + shocks`` with ``beta`` capped to
    ``(-0.9, 0.9)`` so contraction holds. The outcome stays ``(n,)`` while ``X`` is
    ``(n, d_x)`` — the case the old ``torch.randn(X.shape, ...)`` uniqueness start crashed.
    """

    beta = Interval(-0.9, 0.9)
    sigma_sq = Positive()

    def __init__(self, d_x: int = 3, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.gamma = nn.Parameter(torch.zeros(d_x, dtype=torch.float32))

    def constrained_params(self) -> dict[str, torch.Tensor]:
        params = super().constrained_params()
        params["gamma"] = self.gamma
        return params

    def best_response(self, peer_agg, X, shocks):
        p = self.params()
        return p["beta"] * peer_agg + (X @ self.gamma) + shocks


def test_check_model_runs_on_vector_covariate() -> None:
    """A d_x=3 contractive vector-covariate model returns a ModelReport and passes uniqueness."""
    torch.manual_seed(0)
    graph = nx.barabasi_albert_graph(80, 3, seed=4)
    X = torch.randn(graph.number_of_nodes(), 3)
    sub = EgoSubstrate.from_networkx(graph, X, k=2, root_sampler_mode="uniform", seed=0)
    assert sub.d_x == 3

    model = _VectorLinearInMeans(d_x=3, picard_tol=1e-7, picard_max=200)
    with torch.no_grad():
        model._raw_beta.copy_(Interval(-0.9, 0.9).inverse(0.4))
    report = check_model(model, sub, n_probe=80)

    # The d_x>1 solve no longer crashes; the contractive map is admissible.
    assert bool(report)
    assert report["uniqueness"].passed  # the (n,)-shaped second start agrees
    assert abs(report["contraction_modulus"].value - 0.4) < 1e-3  # ||A||_inf = |beta|
    assert report["locality_A2"].passed


def test_check_model_scalar_builtin_rejects_2d_covariate_attributably() -> None:
    """A SCALAR built-in fed a 2-D (d_x>=2) covariate is rejected by check_model with the
    model's attributable 'scalar covariate only' ValueError — not the raw broadcast
    RuntimeError the first Picard solve used to leak before forward's guard was reachable
    (D7-REG-checkmodel-2d-x-raw-error). The vector-covariate framework model and the scalar
    built-in on a 1-D covariate still pass (the guard is exactly forward's own validator)."""
    torch.manual_seed(0)
    graph = nx.barabasi_albert_graph(60, 3, seed=4)
    X_2d = torch.randn(graph.number_of_nodes(), 3)
    sub_2d = EgoSubstrate.from_networkx(graph, X_2d, k=2, root_sampler_mode="uniform", seed=0)
    assert sub_2d.d_x == 3

    for model in (
        LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5),
        EffortGameGenerator(fix_r=1.0, fix_sigma_sq=1.0, init_gamma=1.2, init_lambda=0.7, init_mu=0.3),
    ):
        try:
            check_model(model, sub_2d, n_probe=60)
        except ValueError as exc:
            assert "scalar covariate only" in str(exc)
            assert type(model).__name__ in str(exc)
        else:  # pragma: no cover - guard against a silent contract regression
            raise AssertionError(
                f"{type(model).__name__} on a 2-D covariate must raise an attributable ValueError."
            )

    # A scalar built-in on a 1-D covariate is unaffected: check_model still runs and passes.
    sub_1d = _substrate(graph)
    assert sub_1d.d_x == 1
    report_1d = check_model(
        LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5), sub_1d, n_probe=60
    )
    assert bool(report_1d)

    # A genuine vector-covariate framework model on the SAME 2-D covariate still passes.
    vec = _VectorLinearInMeans(d_x=3, picard_tol=1e-7, picard_max=200)
    with torch.no_grad():
        vec._raw_beta.copy_(Interval(-0.9, 0.9).inverse(0.4))
    report_2d = check_model(vec, sub_2d, n_probe=60)
    assert bool(report_2d)


# ----------------------------------------------------------- interaction-radius locality
class _TwoHopGame(NetworkGameGenerator):
    """A genuine 2-hop game: ``peer_aggregate(W, Y) = W @ (W @ Y)`` (reach radius 2).

    Its Jacobian ``A = beta * W @ W`` is supported on the 2-hop ball, so it is *non-local*
    at interaction radius 1 (``dB_i/dY_j != 0`` for a 2-hop ``j``) but local at radius 2
    (zero outside ``B_2(i)``). ``beta`` capped to ``(-0.9, 0.9)``: ``||W@W||_inf = 1`` (a
    product of row-stochastic matrices is row-stochastic), so the map stays contractive.
    """

    beta = Interval(-0.9, 0.9)
    sigma_sq = Positive()

    def peer_aggregate(self, W, Y):
        once = torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)
        return torch.sparse.mm(W, once.unsqueeze(-1)).squeeze(-1)

    def best_response(self, peer_agg, X, shocks):
        return self.params()["beta"] * peer_agg + shocks


def test_interaction_radius_two_hop_model() -> None:
    """A 2-hop model FAILS locality at r0=1 but PASSES at r0=2; the 1-hop builtin still passes at r0=1."""
    sub = _substrate(nx.barabasi_albert_graph(80, 3, seed=6))
    model = _TwoHopGame()
    with torch.no_grad():
        model._raw_beta.copy_(Interval(-0.9, 0.9).inverse(0.4))

    at_r1 = check_model(model, sub, n_probe=80, interaction_radius=1)
    at_r2 = check_model(model, sub, n_probe=80, interaction_radius=2)

    # Non-local at 1 hop: there is a nonzero 2-hop off-neighbourhood derivative.
    assert not at_r1["locality_A2"].passed
    assert at_r1["locality_A2"].value > at_r1["locality_A2"].threshold
    # Local at 2 hops: the derivative outside B_2(i) is (numerically) zero.
    assert at_r2["locality_A2"].passed
    assert "B_2(i)" in at_r2["locality_A2"].detail

    # The built-in 1-hop model is unchanged: passes at the default radius with the
    # original 1-hop detail string (interaction_radius=1 path preserved).
    builtin = check_model(
        LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5),
        sub, n_probe=80, interaction_radius=1,
    )
    assert builtin["locality_A2"].passed
    assert builtin["locality_A2"].detail == "max |dB_i/dY_j|, j not in 1-hop(i)"


def test_interaction_radius_rejects_invalid() -> None:
    """interaction_radius must be an int >= 1."""
    sub = _substrate(nx.barabasi_albert_graph(40, 2, seed=7))
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.3, init_gamma=1.0)
    for bad in (0, -1):
        try:
            check_model(model, sub, n_probe=40, interaction_radius=bad)
        except ValueError as exc:
            assert "interaction_radius" in str(exc)
        else:  # pragma: no cover - guard against a silent contract regression
            raise AssertionError("interaction_radius < 1 must raise ValueError.")


# ------------------------------------------------------------------ degenerate-probe guard
def test_degenerate_probe_is_surfaced() -> None:
    """beta~0 (no peer effect) flags DEGENERATE on contraction; beta=0.5 does not."""
    sub = _substrate(nx.barabasi_albert_graph(80, 3, seed=8))

    degenerate = check_model(
        LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=1.5), sub, n_probe=80
    )
    contraction = degenerate["contraction_modulus"]
    assert contraction.value < 1e-8  # negligible peer coupling at beta = 0
    assert "DEGENERATE" in contraction.detail
    # A degenerate probe still "passes" the < 1 threshold numerically, but the detail
    # loudly says the certificate is meaningless.
    assert contraction.passed

    active = check_model(
        LinearInMeansGenerator(beta_cap=0.85, init_beta=0.5, init_gamma=1.5), sub, n_probe=80
    )
    assert "DEGENERATE" not in active["contraction_modulus"].detail
    assert active["contraction_modulus"].value > 1e-8


# ------------------------------------------------------- (M) moment-condition observability
def test_estimate_branching_is_finite_positive() -> None:
    """estimate_branching returns a finite positive lambda on a small regular-ish graph."""
    # A 4-regular ring lattice: locally tree-like growth, branching > 1.
    sub = _substrate(nx.watts_strogatz_graph(120, 4, 0.0, seed=9))
    lam = estimate_branching(sub, max_depth=3, n_roots=120, seed=0)
    assert math.isfinite(lam)
    assert lam > 1.0  # a connected non-degenerate graph grows (branching exceeds 1)


def test_estimate_branching_validates_inputs() -> None:
    sub = _substrate(nx.barabasi_albert_graph(40, 2, seed=10))
    for bad_depth in (0, 1):
        try:
            estimate_branching(sub, max_depth=bad_depth)
        except ValueError as exc:
            assert "max_depth" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("max_depth < 2 must raise.")


def test_moment_condition_margin_holds_and_fails() -> None:
    """moment_condition_margin reports holds per rho_bar^p * lambda < 1 with the rate gamma."""
    # rho_bar * lambda^(1/p) = 0.4 * sqrt(2) ~ 0.566 < 1 -> holds; value = 0.16 * 2 = 0.32.
    ok = moment_condition_margin(rho_bar=0.4, lambda_=2.0, p=2.0)
    assert ok["holds"] is True
    assert abs(ok["value"] - 0.4**2 * 2.0) < 1e-12
    # gamma = |log eta| / (2 log lambda + |log eta|), eta = 0.4 * sqrt(2).
    eta = 0.4 * 2.0 ** 0.5
    expected_gamma = abs(math.log(eta)) / (2.0 * math.log(2.0) + abs(math.log(eta)))
    assert math.isfinite(ok["gamma"])
    assert abs(ok["gamma"] - expected_gamma) < 1e-12
    assert 0.0 < ok["gamma"] < 1.0

    # rho_bar^p * lambda = 0.81 * 4 = 3.24 > 1 -> fails (M).
    bad = moment_condition_margin(rho_bar=0.9, lambda_=4.0, p=2.0)
    assert bad["holds"] is False
    assert bad["value"] > 1.0


def test_moment_condition_margin_validates_inputs() -> None:
    for kwargs in (
        {"rho_bar": 0.0, "lambda_": 2.0, "p": 2.0},
        {"rho_bar": 0.4, "lambda_": 0.0, "p": 2.0},
        {"rho_bar": 0.4, "lambda_": 2.0, "p": 0.5},
    ):
        try:
            moment_condition_margin(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid {kwargs} must raise ValueError.")
