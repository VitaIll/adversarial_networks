"""Public re-export of the adversarial game losses.

The canonical implementations live in :mod:`adversarial_networks.core.objective`
(the single source of truth for the criterion math, unit-tested in isolation).
This module re-exports the three losses as a stable, discoverable public surface;
the estimator and any custom training loop call them rather than inlining
``softplus`` arithmetic.
"""

from __future__ import annotations

from .core.objective import (
    discriminator_loss,
    generator_nonsaturating_loss,
    generator_saturating_loss,
)

__all__ = [
    "discriminator_loss",
    "generator_nonsaturating_loss",
    "generator_saturating_loss",
]
