"""End-to-end tests for the AdversarialEstimator engine.

These exercise a small but complete estimation run: build a substrate, simulate an
observed outcome from a known model, then fit a fresh model + discriminator. They
verify the typed result surface, observability collection, that the structural
parameters actually move (gradient flows through the unrolled solve), and that the
milestone-2 gradient-transform seam is invoked with diagnostics propagated to the
step metrics.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
import torch

from src.contracts import EstimationResult, StepMetrics
from src.discriminator import RootedMPNNDiscriminator
from src.ego import EgoSubstrate
from src.estimator import AdversarialEstimator
from src.estimator_config import EstimatorConfig
from src.generator import SCMGenerator
from src.observability import InMemoryHistory


def _build_substrate(n: int = 40, k: int = 2, seed: int = 0) -> EgoSubstrate:
    graph = nx.barabasi_albert_graph(n=n, m=2, seed=seed)
    torch.manual_seed(seed)
    X = torch.randn(graph.number_of_nodes())
    return EgoSubstrate.from_networkx(graph, X, k=k, root_sampler_mode="uniform", seed=seed)


def _true_outcome(substrate: EgoSubstrate, *, beta_cap: float = 0.85) -> torch.Tensor:
    torch.manual_seed(123)
    true_model = SCMGenerator(
        beta_cap=beta_cap, picard_tol=1e-6, picard_max=50,
        init_beta=0.4, init_gamma=1.5, init_log_sigma_sq=0.0,
    )
    with torch.no_grad():
        return true_model(substrate.W, substrate.X)


def _fresh_pair(beta_cap: float = 0.85, hidden_dim: int = 8, k: int = 2):
    model = SCMGenerator(
        beta_cap=beta_cap, picard_tol=1e-6, picard_max=50,
        init_beta=0.0, init_gamma=0.0, init_log_sigma_sq=0.0,
    )
    disc = RootedMPNNDiscriminator(hidden_dim=hidden_dim, num_layers=k, logit_clip=10.0)
    return model, disc


def _small_config(max_steps: int = 15) -> EstimatorConfig:
    return EstimatorConfig(
        max_steps=max_steps, min_steps=0, batch_size=8, n_disc=1,
        lr_d=1e-3, lr_g=5e-3, grad_clip_norm=10.0,
        convergence_window=100, stability_window=30, seed=7,
    )


def test_fit_returns_ok_result_with_model_param_keys() -> None:
    substrate = _build_substrate()
    Y_obs = _true_outcome(substrate)
    model, disc = _fresh_pair()
    history = InMemoryHistory()
    est = AdversarialEstimator(
        model=model, discriminator=disc, substrate=substrate, Y_obs=Y_obs,
        config=_small_config(15), observers=[history],
    )
    result = est.fit()

    assert isinstance(result, EstimationResult)
    assert result.status == "ok"
    assert result.ok
    assert result.n_steps_run == 15
    assert set(result.params.keys()) == {"beta", "gamma", "sigma_sq"}
    assert all(math.isfinite(v) for v in result.params.values())
    assert len(history) == 15
    assert set(history.params.keys()) == {"beta", "gamma", "sigma_sq"}
    assert history.result is result


def test_fit_moves_structural_parameters() -> None:
    """Gradient must flow through the unrolled Picard solve and move the params."""
    substrate = _build_substrate(seed=1)
    Y_obs = _true_outcome(substrate)
    model, disc = _fresh_pair()
    before = model.get_params()
    est = AdversarialEstimator(
        model=model, discriminator=disc, substrate=substrate, Y_obs=Y_obs,
        config=_small_config(20),
    )
    est.fit()
    after = model.get_params()
    moved = max(abs(after[name] - before[name]) for name in before)
    assert moved > 1e-4, f"parameters did not move (max change {moved})"


def test_losses_are_finite_and_positive() -> None:
    substrate = _build_substrate(seed=2)
    Y_obs = _true_outcome(substrate)
    model, disc = _fresh_pair()
    history = InMemoryHistory()
    est = AdversarialEstimator(
        model=model, discriminator=disc, substrate=substrate, Y_obs=Y_obs,
        config=_small_config(12), observers=[history],
    )
    est.fit()
    assert all(math.isfinite(v) and v > 0.0 for v in history.loss_d)
    assert all(math.isfinite(v) and v > 0.0 for v in history.loss_g)


def test_gradient_transform_seam_is_invoked_and_extras_propagate() -> None:
    substrate = _build_substrate(seed=3)
    Y_obs = _true_outcome(substrate)
    model, disc = _fresh_pair()

    seen_steps: list[int] = []

    def transform(estimator: AdversarialEstimator, step: int):
        # The seam must expose the estimation state the M2 preconditioner needs.
        assert estimator.model is model
        assert estimator.W is substrate.W
        seen_steps.append(step)
        return {"probe": 1.0}

    captured: list[StepMetrics] = []

    class _Capture(InMemoryHistory):
        def on_step(self, metrics: StepMetrics) -> None:
            super().on_step(metrics)
            captured.append(metrics)

    est = AdversarialEstimator(
        model=model, discriminator=disc, substrate=substrate, Y_obs=Y_obs,
        config=_small_config(8), observers=[_Capture()], gradient_transform=transform,
    )
    result = est.fit()
    assert seen_steps == list(range(1, result.n_steps_run + 1))
    assert all(m.extras.get("probe") == 1.0 for m in captured)


def test_estimator_rejects_non_conforming_model() -> None:
    substrate = _build_substrate(seed=4)
    Y_obs = _true_outcome(substrate)
    _, disc = _fresh_pair()

    class NotAModel(torch.nn.Module):
        pass

    with pytest.raises(TypeError):
        AdversarialEstimator(
            model=NotAModel(), discriminator=disc, substrate=substrate,  # type: ignore[arg-type]
            Y_obs=Y_obs, config=_small_config(3),
        )


def test_estimator_rejects_wrong_length_outcome() -> None:
    substrate = _build_substrate(seed=5)
    model, disc = _fresh_pair()
    with pytest.raises(ValueError):
        AdversarialEstimator(
            model=model, discriminator=disc, substrate=substrate,
            Y_obs=torch.randn(substrate.num_nodes + 3), config=_small_config(3),
        )
