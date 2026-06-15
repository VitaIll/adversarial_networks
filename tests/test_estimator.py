"""Tests for the sklearn/DoubleML-shaped AdversarialEstimator.

Exercise the public estimator contract: ``fit(data) -> self`` with trailing-
underscore learned attributes, ``NotFittedError`` before fit, clone-safety (the
ctor modules stay pristine), the ``estimates_`` table, that the structural
parameters move (gradient flows through the unrolled solve), the
``MinimaxStepContext`` gradient-transform seam, the receptive-field guard, and the
non-convergence warning.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd
import pytest
import torch

import adversarial_networks as an
from adversarial_networks.contracts import EstimationResult
from adversarial_networks.discriminator import RootedMPNNDiscriminator
from adversarial_networks.estimator import (
    AdversarialEstimator,
    ConvergenceWarning,
    MinimaxStepContext,
    NotFittedError,
)
from adversarial_networks.estimator_config import EstimatorConfig
from adversarial_networks.generators import LinearInMeansGenerator


def _data(n: int = 120, seed: int = 0) -> an.NetworkData:
    return an.make_linear_in_means(n_nodes=n, graph="ba", k=2, seed=seed, m=2)


def _pair(hidden_dim: int = 8, k: int = 2):
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = RootedMPNNDiscriminator(hidden_dim=hidden_dim, num_layers=k, logit_clip=10.0)
    return model, disc


def _small_config(max_steps: int = 15) -> EstimatorConfig:
    return EstimatorConfig(
        max_steps=max_steps, min_steps=0, batch_size=8, n_disc=1, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=100, stability_window=30, seed=7,
    )


def test_fit_returns_self_with_learned_attrs() -> None:
    data = _data()
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returned = est.fit(data)
    assert returned is est
    assert isinstance(est.result_, EstimationResult)
    assert est.result_.status == "ok"
    assert est.n_iter_ == 12
    assert set(est.params_.keys()) == {"beta", "gamma", "sigma_sq"}
    assert est.feature_names_ == ["beta", "gamma", "sigma_sq"]
    assert all(math.isfinite(v) for v in est.params_.values())
    assert len(est.history_) == 12


def test_not_fitted_error_before_fit() -> None:
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(3))
    for attr in ("params_", "model_", "result_", "converged_", "estimates_"):
        with pytest.raises(NotFittedError):
            getattr(est, attr)


def test_clone_safety_ctor_modules_untouched() -> None:
    data = _data(seed=1)
    model, disc = _pair()
    before = model.get_params()
    est = AdversarialEstimator(model, disc, config=_small_config(15))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    assert est.model_ is not model
    assert est.discriminator_ is not disc
    moved = max(abs(est.params_[k] - before[k]) for k in before)
    assert moved > 1e-4, f"learned params did not move ({moved})"
    assert model.get_params() == before  # ctor model pristine (no warm-start)


def test_estimates_table_shape_and_columns() -> None:
    data = _data(seed=2)
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    table = est.estimates_
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["coef", "final", "path_sd"]
    assert list(table.index) == ["beta", "gamma", "sigma_sq"]
    assert (table["path_sd"] >= 0).all()


def test_losses_finite_and_positive() -> None:
    data = _data(seed=3)
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    assert all(math.isfinite(v) and v > 0.0 for v in est.history_.loss_d)
    assert all(math.isfinite(v) and v > 0.0 for v in est.history_.loss_g)


def test_gradient_transform_seam_receives_context_and_extras_propagate() -> None:
    from adversarial_networks.contracts import StepMetrics
    from adversarial_networks.observability import InMemoryHistory

    data = _data(seed=4)
    model, disc = _pair()
    seen: list[int] = []

    def transform(context: MinimaxStepContext):
        assert isinstance(context, MinimaxStepContext)
        assert context.W is context.substrate.W
        assert context.X is context.substrate.X
        assert context.Y_obs.shape[0] == context.substrate.num_nodes
        assert "sigma_X" in context.norm_stats
        seen.append(context.step)
        return {"probe": 2.0}

    captured: list[StepMetrics] = []

    class _Capture(InMemoryHistory):
        def on_step(self, metrics: StepMetrics) -> None:
            super().on_step(metrics)
            captured.append(metrics)

    est = AdversarialEstimator(
        model, disc, config=_small_config(8), gradient_transform=transform, observers=[_Capture()]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    assert seen == list(range(1, est.n_iter_ + 1))
    assert all(m.extras.get("probe") == 2.0 for m in captured)


def test_receptive_field_guard_rejects_shallow_discriminator() -> None:
    data = _data(seed=5)  # k = 2
    model = LinearInMeansGenerator(beta_cap=0.85)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=1)  # < k
    est = AdversarialEstimator(model, disc, config=_small_config(3))
    with pytest.raises(ValueError, match="num_layers"):
        est.fit(data)


def test_unclipped_discriminator_warns_once_about_c6() -> None:
    """logit_clip=None disables the C6(i) bound -> a one-time RuntimeWarning on fit."""
    data = _data(seed=10)
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=None)
    est = AdversarialEstimator(model, disc, config=_small_config(4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(data)
    clip_warnings = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "logit_clip=None" in str(w.message)
    ]
    assert len(clip_warnings) == 1  # surfaced exactly once at the start of the run
    assert "C6" in str(clip_warnings[0].message)


def test_clipped_discriminator_does_not_warn_about_c6() -> None:
    """A positive logit_clip satisfies C6(i): no clip warning is emitted."""
    data = _data(seed=11)
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2)  # default clip 5.0
    est = AdversarialEstimator(model, disc, config=_small_config(4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(data)
    assert not [w for w in caught if "logit_clip=None" in str(w.message)]


def test_fit_rejects_non_networkdata() -> None:
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(3))
    with pytest.raises(TypeError):
        est.fit(object())  # type: ignore[arg-type]


def test_shortcut_kwargs_override_config() -> None:
    data = _data(seed=6)
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(50), max_steps=5, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    assert est.n_iter_ == 5  # the max_steps shortcut won over the config's 50


def test_non_convergence_warns() -> None:
    data = _data(seed=7)
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(6))
    with pytest.warns(ConvergenceWarning):
        est.fit(data)
    assert est.converged_ is False


def test_picard_cap_hit_warns_and_records_in_extras() -> None:
    """A model whose Picard cap is too low to converge feeds a non-equilibrium iterate
    to the discriminator: the engine must warn once about the non-convergence (keyed off
    the solver's converged flag, with the residual surfaced) and record the hit count in
    ``EstimationResult.extras`` (the THEORY_ENFORCEMENT.md claim)."""
    data = _data(n=60, seed=12)
    # picard_max=1 cannot reach the (beta=0.5) equilibrium from the zero start, so every
    # structural-phase forward hits the cap without the tol test firing.
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.5, init_gamma=0.0, picard_max=1)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    est = AdversarialEstimator(model, disc, config=_small_config(4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(data)

    picard_warnings = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "Picard did not converge" in str(w.message)
    ]
    assert len(picard_warnings) == 1  # surfaced exactly once (on the first non-convergence)
    assert "picard_max=1" in str(picard_warnings[0].message)
    assert "residual" in str(picard_warnings[0].message)  # residual surfaced
    assert est.result_.extras["picard_cap_hits"] > 0
    # A closed-form best_response model never runs Newton, so no Newton diagnostic (D6-R2).
    assert "newton_cap_hits" not in est.result_.extras


def test_sampler_shortfall_warns_once() -> None:
    """A disjoint packer that cannot fill the requested batch must surface the shortfall
    exactly once (requested vs achieved + fallback reason), matching the cap-hit style."""
    import networkx as nx

    from adversarial_networks.ego import EgoSubstrate

    graph = nx.barabasi_albert_graph(n=80, m=2, seed=3)
    torch.manual_seed(4)
    X = torch.randn(graph.number_of_nodes(), dtype=torch.float32)
    # A large exclusion radius + large requested batch forces a packing shortfall.
    substrate = EgoSubstrate.from_networkx(
        graph, X, k=2, root_sampler_mode="disjoint_best_of_k",
        exclusion_r=6, disjoint_restarts_k=2, disjoint_fallback="best", seed=1,
    )
    true_model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    torch.manual_seed(5)
    with torch.no_grad():
        y = true_model(substrate.W, substrate.X).detach().to(torch.float32)
    data = an.NetworkData(substrate, y)

    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    cfg = EstimatorConfig(
        max_steps=4, min_steps=0, batch_size=40, n_disc=1, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=100, stability_window=30, seed=7,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AdversarialEstimator(model, disc, config=cfg).fit(data)
    shortfall = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "Root sampler shortfall" in str(w.message)
    ]
    assert len(shortfall) == 1  # surfaced exactly once across all steps
    assert "requested 40" in str(shortfall[0].message)


def test_picard_nonconvergence_flagged_at_default_cap_for_rho095() -> None:
    """A rho~0.95 contraction at the DEFAULT picard cap (200) does not converge at
    tol=1e-6 (paper T ~ 270): the engine must flag non-convergence (keyed off the
    solver's converged flag, NOT iters>=cap) and surface the residual (D2-01)."""
    data = _data(n=60, seed=13)
    # beta_cap=0.99, init_beta=0.95 -> rho~0.95; default picard_max=200 under-solves.
    model = LinearInMeansGenerator(beta_cap=0.99, init_beta=0.95, init_gamma=0.0)
    assert model.picard_max == 200  # base default raised 100 -> 200
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    est = AdversarialEstimator(model, disc, config=_small_config(3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(data)
    picard_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning) and "Picard did not converge" in str(w.message)
    ]
    assert len(picard_warnings) == 1
    assert "residual" in str(picard_warnings[0].message)
    assert est.result_.extras["picard_cap_hits"] > 0


def test_effort_game_fit_reports_newton_cap_hits_key() -> None:
    """An effort-game model ACTUALLY runs Newton, so its extras must carry a
    'newton_cap_hits' key; a closed-form best_response model must not (D6-R2)."""
    eff_data = an.make_effort_game(n_nodes=60, graph="ba", k=2, seed=3, m=2)
    model = an.EffortGameGenerator(init_lambda=0.5, init_mu=0.1)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    est = AdversarialEstimator(model, disc, config=_small_config(4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(eff_data)
    assert "newton_cap_hits" in est.result_.extras  # effort game ran Newton

    # Contrast: the closed-form linear model never runs Newton -> no key.
    lin = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    lin_disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    lin_est = AdversarialEstimator(lin, lin_disc, config=_small_config(4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin_est.fit(_data(seed=14))
    assert "newton_cap_hits" not in lin_est.result_.extras


def test_step_metrics_carry_picard_and_sampler_fields() -> None:
    """StepMetrics surface the per-step picard residual/converged and structural-phase
    sampler met_target/fallback_reason (D2-01, D4-05)."""
    from adversarial_networks.contracts import StepMetrics
    from adversarial_networks.observability import InMemoryHistory

    captured: list[StepMetrics] = []

    class _Capture(InMemoryHistory):
        def on_step(self, m: StepMetrics) -> None:
            super().on_step(m)
            captured.append(m)

    data = _data(seed=15)
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(4), observers=[_Capture()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    assert captured
    m = captured[-1]
    # converged forward (uniform sampler fills the batch)
    assert m.picard_converged is True
    assert m.picard_residual < model.picard_tol
    assert m.sampler_met_target is True
    assert m.sampler_fallback_reason == ""


def test_step_metrics_carry_non_default_sampler_fallback_reason() -> None:
    """Under a structural-phase uniform fallback, a per-step StepMetrics must carry the
    NON-default sampler_fallback_reason (not the dataclass default ''), proving the field
    is wired to root_result rather than coinciding with the happy-path default (D4-REG).

    Asserts on sampler_fallback_reason, NOT sampler_met_target: the uniform-fallback path
    sets met_target=True on the uniform draw (sampling.py _uniform_result), so only
    fallback_reason diverges from the default here.
    """
    import networkx as nx

    from adversarial_networks.contracts import StepMetrics
    from adversarial_networks.ego import EgoSubstrate
    from adversarial_networks.observability import InMemoryHistory

    # path graph (sparse) + disjoint_relax with a single radius=2*k=4 rung whose packing
    # cannot reach disjoint_min_batch -> the 'uniform' fallback fires for the structural draw.
    graph = nx.path_graph(60)
    torch.manual_seed(0)
    X = torch.randn(60, dtype=torch.float32)
    substrate = EgoSubstrate.from_networkx(
        graph, X, k=2, root_sampler_mode="disjoint_relax",
        disjoint_relax_sequence=(4,), disjoint_min_batch=30, disjoint_restarts_k=3,
        disjoint_fallback="uniform", seed=1,
    )
    true_model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    torch.manual_seed(5)
    with torch.no_grad():
        y = true_model(substrate.W, substrate.X).detach().to(torch.float32)
    data = an.NetworkData(substrate, y)

    captured: list[StepMetrics] = []

    class _Capture(InMemoryHistory):
        def on_step(self, m: StepMetrics) -> None:
            super().on_step(m)
            captured.append(m)

    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    cfg = EstimatorConfig(
        max_steps=3, min_steps=0, batch_size=30, n_disc=1, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=100, stability_window=30, seed=7,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        AdversarialEstimator(model, disc, config=cfg, observers=[_Capture()]).fit(data)

    assert captured
    # Every structural draw fell back to uniform here, so the per-step field is non-default.
    assert all(
        m.sampler_fallback_reason.startswith("uniform_fallback_below_min_batch")
        for m in captured
    ), [m.sampler_fallback_reason for m in captured]


def test_step_metrics_carry_picard_nonconvergence_fields() -> None:
    """Under a forced Picard cap, a per-step StepMetrics must carry the NON-default
    picard_converged=False with picard_residual>0 (not the dataclass defaults
    picard_converged=True / picard_residual=0.0), proving those fields are wired to
    root_result/the solver rather than coinciding with the happy path (D4-REG)."""
    from adversarial_networks.contracts import StepMetrics
    from adversarial_networks.observability import InMemoryHistory

    captured: list[StepMetrics] = []

    class _Capture(InMemoryHistory):
        def on_step(self, m: StepMetrics) -> None:
            super().on_step(m)
            captured.append(m)

    data = _data(n=60, seed=12)
    # picard_max=1 from a zero start cannot reach the beta=0.5 equilibrium -> every
    # structural-phase forward hits the cap without the tolerance test firing.
    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.5, init_gamma=0.0, picard_max=1)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    est = AdversarialEstimator(model, disc, config=_small_config(3), observers=[_Capture()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)

    assert captured
    assert all(m.picard_converged is False for m in captured)
    assert all(m.picard_residual > 0.0 for m in captured)


def test_disc_phase_sampler_shortfall_warns_once() -> None:
    """A disjoint packer that under-fills the DISCRIMINATOR-phase batch must surface a
    one-time disc-phase shortfall warning (today only the structural phase warned) (D4-05).
    """
    import networkx as nx

    from adversarial_networks.ego import EgoSubstrate

    graph = nx.barabasi_albert_graph(n=80, m=2, seed=3)
    torch.manual_seed(4)
    X = torch.randn(graph.number_of_nodes(), dtype=torch.float32)
    substrate = EgoSubstrate.from_networkx(
        graph, X, k=2, root_sampler_mode="disjoint_best_of_k",
        exclusion_r=6, disjoint_restarts_k=2, disjoint_fallback="best", seed=1,
    )
    true_model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    torch.manual_seed(5)
    with torch.no_grad():
        y = true_model(substrate.W, substrate.X).detach().to(torch.float32)
    data = an.NetworkData(substrate, y)

    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    cfg = EstimatorConfig(
        max_steps=4, min_steps=0, batch_size=40, n_disc=2, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=100, stability_window=30, seed=7,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AdversarialEstimator(model, disc, config=cfg).fit(data)
    disc_shortfall = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "Discriminator-phase root sampler shortfall" in str(w.message)
    ]
    assert len(disc_shortfall) == 1


def test_sub_two_k_radius_warns_once() -> None:
    """A disjoint_relax ladder forced to step down to a sub-2k rung that still FILLS the
    batch forfeits vertex-disjointness silently: the engine must surface a below-2k
    warning keyed on the independence property, not batch-fill (D4-06)."""
    import networkx as nx

    from adversarial_networks.ego import EgoSubstrate

    # path graph, k=2 -> 2k=4. Relax ladder top rung 4 (vertex-disjoint) steps down to 2
    # (< 2k) to fill the batch.
    graph = nx.path_graph(60)
    torch.manual_seed(0)
    X = torch.randn(60, dtype=torch.float32)
    substrate = EgoSubstrate.from_networkx(
        graph, X, k=2, root_sampler_mode="disjoint_relax",
        disjoint_relax_sequence=(4, 2), disjoint_min_batch=13, disjoint_restarts_k=4,
        disjoint_fallback="best", seed=1,
    )
    true_model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.4, init_gamma=1.5)
    torch.manual_seed(5)
    with torch.no_grad():
        y = true_model(substrate.W, substrate.X).detach().to(torch.float32)
    data = an.NetworkData(substrate, y)

    model = LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    cfg = EstimatorConfig(
        max_steps=3, min_steps=0, batch_size=13, n_disc=1, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=100, stability_window=30, seed=7,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AdversarialEstimator(model, disc, config=cfg).fit(data)
    sub_2k = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning) and "< 2*k" in str(w.message)
    ]
    assert len(sub_2k) == 1


def test_residual_blur_guard_fires_at_tail_window_start() -> None:
    """The residual-blur guard fires when the blur is still > 0 at the START of the
    tail-averaging window, even if it reaches zero by max_steps — the point estimate is
    the tail average, so a contaminated tail must be flagged (D1-03R)."""
    from adversarial_networks.config import InstanceNoiseConfig

    data = _data(seed=16)
    model, disc = _pair()
    # max_steps=40, tail_window=max(conv=30, stab=20)=30 -> tail starts at step 11.
    # anneal_steps=35 reaches zero by step 40 (terminal) BUT tau>0 at step 11.
    cfg = EstimatorConfig(
        max_steps=40, min_steps=0, batch_size=8, n_disc=1, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=30, stability_window=20, seed=7,
    )
    blur = InstanceNoiseConfig(enabled=True, tau_y0=1.0, schedule="linear", anneal_steps=35, min_tau=0.0)
    est = AdversarialEstimator(model, disc, config=cfg, instance_noise=blur)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(data)
    blur_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "does not reach zero by the start of the tail-averaging window" in str(w.message)
    ]
    assert len(blur_warnings) == 1

    # A blur that reaches zero before the tail-window start emits NO warning.
    model2, disc2 = _pair()
    blur_ok = InstanceNoiseConfig(enabled=True, tau_y0=1.0, schedule="linear", anneal_steps=10, min_tau=0.0)
    est2 = AdversarialEstimator(model2, disc2, config=cfg, instance_noise=blur_ok)
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        est2.fit(data)
    assert not [w for w in caught2 if "tail-averaging window" in str(w.message)]


def test_exp_schedule_residual_blur_warns_and_points_to_linear() -> None:
    """While annealing (step < anneal_steps), the 'exp' schedule only asymptotes toward
    min_tau and so leaves residual blur in the tail-averaging window; the guard fires and
    the message must point to the LINEAR schedule and explain exp cannot reach exactly zero
    while annealing (D1-03R-exp / D6-R1). (Once step >= anneal_steps the exp branch snaps to
    exactly min_tau, mirroring linear — D3-REG-exp-config-guard-strict-zero — so this uses
    anneal_steps reaching INTO the tail window to keep residual blur at the window start.)"""
    from adversarial_networks.config import InstanceNoiseConfig

    data = _data(seed=17)
    model, disc = _pair()
    cfg = EstimatorConfig(
        max_steps=40, min_steps=0, batch_size=8, n_disc=1, lr_d=1e-3, lr_g=5e-3,
        grad_clip_norm=10.0, convergence_window=30, stability_window=20, seed=7,
    )
    # tail_window=max(30,20)=30 -> tail starts at step 11; anneal_steps=40 keeps the exp
    # blur strictly positive at step 11 (asymptotic, not yet snapped), so the guard fires.
    blur_exp = InstanceNoiseConfig(enabled=True, tau_y0=1.0, schedule="exp", anneal_steps=40, min_tau=0.0)
    est = AdversarialEstimator(model, disc, config=cfg, instance_noise=blur_exp)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(data)
    exp_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning) and "tail-averaging window" in str(w.message)
    ]
    assert len(exp_warnings) == 1
    msg = str(exp_warnings[0].message)
    assert "linear schedule" in msg  # remedy directs to the linear schedule
    assert "exp" in msg and "never reaches exactly zero" in msg  # explains exp asymptotes


def test_simulate_and_discriminator_scores() -> None:
    data = _data(seed=8)
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    y_sim = est.simulate(seed=1)
    assert y_sim.shape == (data.num_nodes,)
    real, fake = est.discriminator_scores(n_roots=32)
    assert real.numel() == 32 and fake.numel() == 32
    assert torch.all((real >= 0) & (real <= 1)) and torch.all((fake >= 0) & (fake <= 1))


def test_recovery_table_against_truth() -> None:
    data = _data(seed=9)
    model, disc = _pair()
    est = AdversarialEstimator(model, disc, config=_small_config(8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    table = est.recovery_table({"beta": 0.4, "gamma": 1.5, "sigma_sq": 1.0})
    assert list(table.columns) == ["coef", "true", "abs_err", "path_sd"]
    assert table.loc["beta", "true"] == 0.4
