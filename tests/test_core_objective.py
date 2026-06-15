"""Deterministic CPU tests for ``core.objective`` (blur schedule + convergence band).

The instance-noise schedule check was relocated from the retired ``test_utils.py``;
it now exercises the kernel under its real name and in its single-return
(outcome-only) form. The convergence-band tests pin the saturating/non-saturating
generator optima the loss-band stopping check compares against.
"""

from __future__ import annotations

import math

from adversarial_networks.config import InstanceNoiseConfig
from adversarial_networks.core.objective import (
    OPTIMAL_DISC_LOSS,
    OPTIMAL_GEN_LOSS,
    OPTIMAL_GEN_LOSS_SATURATING,
    check_gan_convergence,
    instance_noise_taus,
)


def test_compute_instance_noise_taus_schedule_variants() -> None:
    """Scheduler returns the expected outcome blur for constant/linear/exp modes."""
    cfg_constant = InstanceNoiseConfig(
        enabled=True,
        tau_y0=0.08,
        schedule="constant",
        anneal_steps=200,
        min_tau=0.0,
    )
    assert instance_noise_taus(cfg_constant, generator_step=123) == 0.08

    cfg_linear = InstanceNoiseConfig(
        enabled=True,
        tau_y0=0.20,
        schedule="linear",
        anneal_steps=100,
        min_tau=0.02,
    )
    assert instance_noise_taus(cfg_linear, generator_step=0) == 0.20
    assert abs(instance_noise_taus(cfg_linear, generator_step=50) - 0.10) < 1e-12
    # Past anneal_steps the linear schedule sits at the min_tau floor.
    assert instance_noise_taus(cfg_linear, generator_step=1000) == 0.02

    cfg_exp = InstanceNoiseConfig(
        enabled=True,
        tau_y0=0.15,
        schedule="exp",
        anneal_steps=100,
        min_tau=0.01,
    )
    expected_ty = max(0.01, 0.15 * math.exp(-20.0 / (100.0 / 5.0)))
    assert abs(instance_noise_taus(cfg_exp, generator_step=20) - expected_ty) < 1e-12


def test_linear_schedule_reaches_exactly_zero_when_min_tau_zero() -> None:
    """With ``min_tau=0`` the linear schedule hits EXACTLY 0.0 at/after anneal_steps.

    This is the consistency property the paper requires: as ``sigma -> 0`` the
    perturbed criterion converges to the original criterion, so a run whose horizon
    reaches ``anneal_steps`` targets the unblurred objective on its tail.
    """
    cfg = InstanceNoiseConfig(
        enabled=True,
        tau_y0=1.0,
        schedule="linear",
        anneal_steps=1000,
        min_tau=0.0,
    )
    assert instance_noise_taus(cfg, generator_step=1000) == 0.0
    assert instance_noise_taus(cfg, generator_step=1001) == 0.0
    assert instance_noise_taus(cfg, generator_step=5000) == 0.0
    # Strictly positive strictly before anneal_steps.
    assert instance_noise_taus(cfg, generator_step=999) > 0.0


def test_exp_schedule_snaps_to_exactly_zero_at_anneal_steps() -> None:
    """With ``min_tau=0`` the exp schedule snaps to EXACTLY 0.0 once
    ``step >= anneal_steps`` (mirroring the linear branch), so a finished exp anneal
    reaches the original criterion and the strict config guard accepts it
    (D3-REG-exp-config-guard-strict-zero). Strictly before anneal_steps it is positive
    (only asymptotic), reflecting that the continuous exp decay alone never hits zero.
    """
    cfg = InstanceNoiseConfig(
        enabled=True,
        tau_y0=1.0,
        schedule="exp",
        anneal_steps=50,
        min_tau=0.0,
    )
    assert instance_noise_taus(cfg, generator_step=50) == 0.0
    assert instance_noise_taus(cfg, generator_step=51) == 0.0
    assert instance_noise_taus(cfg, generator_step=800) == 0.0
    # Strictly before anneal_steps the exp blur is strictly positive (asymptotic decay).
    assert instance_noise_taus(cfg, generator_step=49) > 0.0


def test_exp_schedule_snaps_to_min_tau_floor_at_anneal_steps() -> None:
    """With a positive ``min_tau`` the exp snap lands EXACTLY on the floor at
    ``step >= anneal_steps`` (not the asymptotic value just above it)."""
    cfg = InstanceNoiseConfig(
        enabled=True,
        tau_y0=1.0,
        schedule="exp",
        anneal_steps=30,
        min_tau=0.05,
    )
    assert instance_noise_taus(cfg, generator_step=30) == 0.05
    assert instance_noise_taus(cfg, generator_step=120) == 0.05


def test_instance_noise_taus_disabled_or_none_returns_zero() -> None:
    """A ``None`` or disabled config yields zero blur (single float)."""
    assert instance_noise_taus(None, generator_step=10) == 0.0
    cfg_off = InstanceNoiseConfig(enabled=False, tau_y0=1.0)
    assert instance_noise_taus(cfg_off, generator_step=10) == 0.0


def test_check_gan_convergence_saturating_vs_nonsaturating_band() -> None:
    """The generator band tracks ``gen_optimum``: -log2 for saturating, +log2 default.

    A discriminator history at ``2 log 2`` plus a generator history at ``-log 2``
    must be accepted under ``gen_optimum=OPTIMAL_GEN_LOSS_SATURATING`` and rejected
    under the (default) non-saturating ``+log 2`` target, and vice-versa.
    """
    window = 20
    loss_d = [OPTIMAL_DISC_LOSS] * window

    loss_g_sat = [OPTIMAL_GEN_LOSS_SATURATING] * window
    loss_g_ns = [OPTIMAL_GEN_LOSS] * window

    # Saturating generator mean (~ -log2): accepted only by the saturating target.
    conv_sat, _, roll_g_sat = check_gan_convergence(
        loss_d, loss_g_sat, window=window, delta_d=0.01, delta_g=0.01,
        gen_optimum=OPTIMAL_GEN_LOSS_SATURATING,
    )
    assert conv_sat is True
    assert abs(roll_g_sat - OPTIMAL_GEN_LOSS_SATURATING) < 1e-9
    conv_sat_wrong, _, _ = check_gan_convergence(
        loss_d, loss_g_sat, window=window, delta_d=0.01, delta_g=0.01,
        gen_optimum=OPTIMAL_GEN_LOSS,
    )
    assert conv_sat_wrong is False

    # Non-saturating generator mean (~ +log2): accepted only by the default target.
    conv_ns, _, _ = check_gan_convergence(
        loss_d, loss_g_ns, window=window, delta_d=0.01, delta_g=0.01,
    )
    assert conv_ns is True
    conv_ns_wrong, _, _ = check_gan_convergence(
        loss_d, loss_g_ns, window=window, delta_d=0.01, delta_g=0.01,
        gen_optimum=OPTIMAL_GEN_LOSS_SATURATING,
    )
    assert conv_ns_wrong is False
