"""Tests for the adversarial game losses.

These pin the losses to (a) their theoretical equilibrium values (``2 log 2`` and
``log 2`` at ``D ≡ 1/2``), (b) the exact reference formula used in the original
training script, (c) boundary-check behaviour, and (d) the non-saturating
gradient property that motivates eq. (4.2).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from src.losses import (
    discriminator_loss,
    generator_nonsaturating_loss,
    generator_saturating_loss,
)


def test_discriminator_loss_at_zero_logits_equals_2log2() -> None:
    real = torch.zeros(8)
    fake = torch.zeros(8)
    loss = discriminator_loss(real, fake)
    assert abs(loss.item() - 2.0 * math.log(2.0)) < 1e-6


def test_generator_nonsaturating_loss_at_zero_logits_equals_log2() -> None:
    fake = torch.zeros(8)
    loss = generator_nonsaturating_loss(fake)
    assert abs(loss.item() - math.log(2.0)) < 1e-6


def test_generator_saturating_loss_at_zero_logits_equals_neg_log2() -> None:
    fake = torch.zeros(8)
    loss = generator_saturating_loss(fake)
    assert abs(loss.item() - (-math.log(2.0))) < 1e-6


def test_discriminator_loss_matches_reference_formula() -> None:
    """Must reproduce the exact softplus formula used in the MC training script."""
    torch.manual_seed(0)
    real = torch.randn(16)
    fake = torch.randn(16)
    reference = F.softplus(-real).mean() + F.softplus(fake).mean()
    assert torch.allclose(discriminator_loss(real, fake), reference)


def test_generator_nonsaturating_loss_matches_reference_formula() -> None:
    torch.manual_seed(1)
    fake = torch.randn(16)
    reference = F.softplus(-fake).mean()
    assert torch.allclose(generator_nonsaturating_loss(fake), reference)


def test_discriminator_loss_accepts_unequal_batch_sizes() -> None:
    real = torch.zeros(5)
    fake = torch.zeros(9)
    loss = discriminator_loss(real, fake)
    assert torch.isfinite(loss)


def test_nonsaturating_gradient_informative_when_discriminator_confident() -> None:
    """Non-saturating loss keeps an informative gradient when D(fake) -> 0.

    With strongly negative fake logits the saturating loss gradient collapses,
    whereas the non-saturating gradient stays O(1); this is the property that
    motivates using eq. (4.2) for the structural update.
    """
    logit_value = -12.0
    fake_ns = torch.full((4,), logit_value, requires_grad=True)
    generator_nonsaturating_loss(fake_ns).backward()
    grad_ns = fake_ns.grad.abs().sum().item()

    fake_sat = torch.full((4,), logit_value, requires_grad=True)
    generator_saturating_loss(fake_sat).backward()
    grad_sat = fake_sat.grad.abs().sum().item()

    assert grad_ns > 0.5, f"non-saturating gradient unexpectedly small: {grad_ns}"
    assert grad_sat < 1e-3, f"saturating gradient unexpectedly large: {grad_sat}"
    assert grad_ns > grad_sat


@pytest.mark.parametrize("bad", [torch.zeros(0), torch.zeros(2, 2), torch.zeros(3, dtype=torch.long)])
def test_generator_loss_rejects_invalid_logits(bad: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        generator_nonsaturating_loss(bad)


def test_discriminator_loss_rejects_non_tensor() -> None:
    with pytest.raises(TypeError):
        discriminator_loss([0.0, 1.0], torch.zeros(2))  # type: ignore[arg-type]
