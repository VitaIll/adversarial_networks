"""Typing-only structural types for the fast computational core.

This module carries *typing* surfaces that the core kernels reference without
incurring a runtime dependency. Keeping them here lets ``core.objective`` and
``core.ego_features`` annotate their instance-noise argument without importing the
workflow-layer ``config`` module (which would invert the dependency arrow).
"""

from __future__ import annotations

from typing import Protocol


class InstanceNoiseConfigLike(Protocol):
    """Structural type for an optional instance-noise (discriminator blur) config.

    Any object exposing these attributes can drive the instance-noise schedule;
    the concrete :class:`adversarial_networks.config.InstanceNoiseConfig` is one
    such object, but the core never imports it.
    """

    enabled: bool
    tau_y0: float
    schedule: str
    anneal_steps: int
    min_tau: float
