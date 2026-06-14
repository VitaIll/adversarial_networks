"""Deterministic CPU tests for ``core.objective.instance_noise_taus``.

Relocated from the retired ``test_utils.py`` (the instance-noise schedule check),
now exercising the kernel under its real name.
"""

from __future__ import annotations

import math

from adversarial_networks.config import InstanceNoiseConfig
from adversarial_networks.core.objective import instance_noise_taus


def test_compute_instance_noise_taus_schedule_variants() -> None:
    """Scheduler returns expected tau values for constant/linear/exp modes."""
    cfg_constant = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.03,
        tau_y0=0.08,
        schedule="constant",
        anneal_steps=200,
        min_tau=0.0,
    )
    tx_c, ty_c = instance_noise_taus(cfg_constant, generator_step=123)
    assert tx_c == 0.03
    assert ty_c == 0.08

    cfg_linear = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.10,
        tau_y0=0.20,
        schedule="linear",
        anneal_steps=100,
        min_tau=0.02,
    )
    tx_0, ty_0 = instance_noise_taus(cfg_linear, generator_step=0)
    tx_mid, ty_mid = instance_noise_taus(cfg_linear, generator_step=50)
    tx_end, ty_end = instance_noise_taus(cfg_linear, generator_step=1000)
    assert tx_0 == 0.10 and ty_0 == 0.20
    assert abs(tx_mid - 0.05) < 1e-12
    assert abs(ty_mid - 0.10) < 1e-12
    assert tx_end == 0.02 and ty_end == 0.02

    cfg_exp = InstanceNoiseConfig(
        enabled=True,
        tau_x0=0.09,
        tau_y0=0.15,
        schedule="exp",
        anneal_steps=100,
        min_tau=0.01,
    )
    tx_e, ty_e = instance_noise_taus(cfg_exp, generator_step=20)
    expected_tx = max(0.01, 0.09 * math.exp(-20.0 / (100.0 / 5.0)))
    expected_ty = max(0.01, 0.15 * math.exp(-20.0 / (100.0 / 5.0)))
    assert abs(tx_e - expected_tx) < 1e-12
    assert abs(ty_e - expected_ty) < 1e-12
