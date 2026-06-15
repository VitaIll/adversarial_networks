"""Slow end-to-end recovery test (a moderate-precision regression guard).

Runs the calibrated recovery recipe (``EstimatorConfig.recovery_default()`` + the
instance-noise blur + the calibrated model/discriminator) on synthetic
linear-in-means data at the ~10k scale and checks that the estimated ``beta``/``gamma``
land in a **moderate-precision band** calibrated to the verified seed-0 reference run
(``beta~0.381``, ``gamma~1.427``; truth ``beta=0.4``, ``gamma=1.5``). The band is a
regression guard pinned to that reference point estimate — tight enough to catch a real
drift, with deliberate robustness margin left so it survives benign torch/platform
variation in this stochastic GNN fit (it is NOT a sampling-uncertainty / standard-error
claim). ``sigma^2`` is biased at finite ``n`` and is not asserted (per the design).

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
        enabled=True, tau_y0=1.0, schedule="linear", anneal_steps=1000,
        min_tau=0.0,
    )
    est = an.AdversarialEstimator(
        model, disc, config=an.EstimatorConfig.recovery_default(), instance_noise=blur
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)

    beta, gamma = est.params_["beta"], est.params_["gamma"]
    # Moderate-precision band calibrated to the verified seed-0 reference run under the
    # paper-faithful criterion (outcomes-only blur annealed to zero, [eta,1-eta]-clipped
    # per-object discriminator, lr decay): seed 0 lands beta~0.381 / gamma~1.427 (truth
    # beta=0.4 / gamma=1.5). The bounds are centered on that reference point estimate with
    # robustness margin left for benign torch/platform variation in this stochastic GNN fit —
    # a regression guard pinned to the seed-0 reference, NOT a sampling-uncertainty / SE claim.
    assert abs(beta - 0.38) < 0.10, f"beta {beta:.3f} outside the seed-0 recovery band (0.28, 0.48)"
    assert abs(gamma - 1.43) < 0.22, f"gamma {gamma:.3f} outside the seed-0 recovery band (1.21, 1.65)"
    # the estimates moved substantially from the init (beta=gamma=0) toward the truth
    assert beta > 0.10 and gamma > 1.00
