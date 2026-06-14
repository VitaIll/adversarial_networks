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
