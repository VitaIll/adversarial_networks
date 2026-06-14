"""Slow end-to-end recovery test (the calibration gate).

Runs the calibrated recovery recipe (``EstimatorConfig.recovery_default()`` + the
instance-noise blur + the calibrated model/discriminator) on synthetic
linear-in-means data at the ~10k scale and asserts the estimated ``beta``/``gamma``
land within the **observed** fast-scale spread — a regression guard, not a precision
claim. ``sigma^2`` is biased at finite ``n`` and is not asserted (per the design).

Marked ``slow``: deselected by default (``pytest.ini`` adds ``-m "not slow"``); run
with ``pytest -m slow``.
"""

from __future__ import annotations

import warnings

import pytest

import adversarial_networks as an
from adversarial_networks.config import InstanceNoiseConfig


@pytest.mark.slow
def test_linear_in_means_recovery_within_observed_spread() -> None:
    data = an.make_linear_in_means(n_nodes=10_000, graph="ba", k=2, seed=0, m=2)
    model = an.LinearInMeansGenerator(beta_cap=0.85)
    disc = an.RootedMPNNDiscriminator(hidden_dim=12, num_layers=2, logit_clip=4.0)
    blur = InstanceNoiseConfig(
        enabled=True, tau_x0=1.0, tau_y0=1.0, schedule="linear", anneal_steps=2000,
        min_tau=0.0, apply_to="both",
    )
    est = an.AdversarialEstimator(
        model, disc, config=an.EstimatorConfig.recovery_default(), instance_noise=blur
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)

    beta, gamma = est.params_["beta"], est.params_["gamma"]
    # Calibrated fast-scale spread (true beta=0.4, gamma=1.5): beta is biased low
    # (the social multiplier is the hardest to identify at fast scale), gamma near 1.5.
    assert 0.10 < beta < 0.50, f"beta {beta:.3f} outside the calibrated recovery band"
    assert 1.10 < gamma < 1.90, f"gamma {gamma:.3f} outside the calibrated recovery band"
    # the estimates moved substantially from the init (beta=gamma=0)
    assert beta > 0.10 and gamma > 1.00
