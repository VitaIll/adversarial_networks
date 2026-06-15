"""Fast, default-suite estimation-accuracy guard (Thm 2 / Thm 13 consistency).

The only end-to-end accuracy test, :mod:`tests.test_recovery`, is marked ``slow`` and
runs at the ~10k scale, so the *default* CI never exercises the central consistency
claim — that the estimator actually drives the structural parameters toward the truth.
This module adds a small, deterministic, **default-run** regression guard: a few-hundred
-step fit on a ~700-node Barabasi-Albert linear-in-means problem at a fixed seed.

It asserts a robust *directional* consistency property — the tail-averaged distance
``|theta_hat - theta_0|`` is strictly smaller than the ``|theta_init - theta_0|`` at the
``beta = gamma = 0`` initialisation (the parameters move toward the truth) — plus a loose
neighbourhood on ``beta``/``gamma`` that still fails if recovery is broken (params stuck at
the init or diverging). The bands are calibrated to be robust at the chosen seed (verified
across 8+ seeds during calibration: ``beta in [0.19, 0.57]``, ``gamma in [1.19, 1.58]``)
while remaining a genuine recovery check, not a tautology.

The fit is fully reproducible: the estimator reseeds the global torch RNG and the
substrate sampler at ``fit`` start, so the result is bit-identical regardless of test
order (``pytest-randomly``) — confirmed not flaky under ambient-RNG perturbation.
"""

from __future__ import annotations

import math
import warnings

import adversarial_networks as an
from adversarial_networks.config import InstanceNoiseConfig
from adversarial_networks.estimator_config import EstimatorConfig

_TRUE = {"beta": 0.4, "gamma": 1.5}
_INIT = {"beta": 0.0, "gamma": 0.0}


def _fast_recovery_config(seed: int) -> EstimatorConfig:
    """A short, calibrated recovery recipe (~280 steps) that runs in a few seconds.

    The blur (``anneal_steps=170``) reaches exactly zero well before the tail-averaging
    window starts (``max_steps - max(convergence_window, stability_window) + 1 = 246``),
    so the estimator targets the original, unblurred criterion on the whole tail (the
    consistency target) and no residual-blur warning fires. ``lr_g`` is decayed at two
    milestones so the optimiser settles as the discriminator sharpens late in training.
    """
    return EstimatorConfig(
        max_steps=280,
        min_steps=0,
        batch_size=16,
        n_disc=1,
        lr_d=4e-4,
        lr_g=8e-3,
        grad_clip_norm=10.0,
        lr_g_decay_steps=(160, 220),
        lr_g_decay_factor=0.5,
        convergence_window=35,
        stability_window=18,
        seed=seed,
    )


def test_fast_linear_in_means_recovery_moves_toward_truth() -> None:
    """A short fit must move ``(beta, gamma)`` from the ``0`` init toward ``(0.4, 1.5)``.

    Primary (robust) assertion: the tail-averaged total parameter error is strictly
    smaller than the error at the ``beta = gamma = 0`` initialisation. Secondary: ``beta``
    and ``gamma`` are finite and land in a loose neighbourhood of the truth (so a broken
    estimator that leaves the params at the init, or diverges, fails this test).
    """
    seed = 0
    data = an.make_linear_in_means(n_nodes=700, graph="ba", k=2, seed=seed, m=2)
    model = an.LinearInMeansGenerator(beta_cap=0.85, init_beta=0.0, init_gamma=0.0)
    disc = an.RootedMPNNDiscriminator(hidden_dim=12, num_layers=2, logit_clip=4.0)
    blur = InstanceNoiseConfig(
        enabled=True, tau_y0=1.0, schedule="linear", anneal_steps=170, min_tau=0.0
    )
    est = an.AdversarialEstimator(
        model, disc, config=_fast_recovery_config(seed), instance_noise=blur
    )
    # The fit must be clean: with the blur annealed to zero before the tail window, the
    # residual-blur RuntimeWarning must NOT fire (any such warning is a real defect here).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(data)
    assert not [
        w for w in caught if "tail-averaging window" in str(w.message)
    ], "residual-blur warning fired despite a clean anneal-to-zero before the tail window"

    beta = est.params_["beta"]
    gamma = est.params_["gamma"]
    assert math.isfinite(beta) and math.isfinite(gamma)

    init_dist = abs(_INIT["beta"] - _TRUE["beta"]) + abs(_INIT["gamma"] - _TRUE["gamma"])
    tail_dist = abs(beta - _TRUE["beta"]) + abs(gamma - _TRUE["gamma"])
    # Directional consistency: the estimate is strictly closer to the truth than the init.
    assert tail_dist < init_dist, (
        f"params did not move toward truth: tail_dist={tail_dist:.3f} "
        f"not < init_dist={init_dist:.3f} (beta={beta:.3f}, gamma={gamma:.3f})"
    )
    # Each parameter individually moved toward its truth (not just net).
    assert abs(beta - _TRUE["beta"]) < abs(_INIT["beta"] - _TRUE["beta"]), (
        f"beta {beta:.3f} did not move toward {_TRUE['beta']} from the 0 init"
    )
    assert abs(gamma - _TRUE["gamma"]) < abs(_INIT["gamma"] - _TRUE["gamma"]), (
        f"gamma {gamma:.3f} did not move toward {_TRUE['gamma']} from the 0 init"
    )
    # Loose neighbourhood of the truth (robust at seed 0; calibration spread over 8 seeds
    # was beta in [0.19, 0.57], gamma in [1.19, 1.58]). Wide enough to absorb seed spread,
    # tight enough to fail if recovery is broken (e.g. beta stuck near 0 / gamma near 0).
    assert 0.15 < beta < 0.70, f"beta {beta:.3f} outside the loose recovery neighbourhood"
    assert 1.00 < gamma < 2.00, f"gamma {gamma:.3f} outside the loose recovery neighbourhood"
