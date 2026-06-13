"""Adversarial objective functions for the rooted ego-object minimax game.

These implement the two losses of the alternating estimator (Algorithm 1):

* :func:`discriminator_loss` — the adaptive test function (discriminator) is
  trained to *maximise* the minibatch criterion (eq. 4.1). Written as a loss to
  *minimise*, it is the binary cross-entropy

      L_D = E[-log D(S_real)] + E[-log(1 - D(S_fake))],

  which, in logit form with ``D = sigmoid(logit)``, equals
  ``softplus(-logit_real).mean() + softplus(logit_fake).mean()``.

* :func:`generator_nonsaturating_loss` — the structural parameters are updated on
  the *non-saturating* generator loss (eq. 4.2),

      L_G = E[-log D(S_fake)] = softplus(-logit_fake).mean(),

  whose gradient with respect to the outcome coordinates remains informative even
  when the discriminator is confident (``D(S_fake) -> 0``); see the paper's
  discussion around eq. (4.2) and footnotes 21-22.

At the population optimum the test function cannot beat random classification,
``D* ≡ 1/2``, so ``L_D -> 2 log 2`` and ``L_G -> log 2`` (see
:data:`src.constants.OPTIMAL_DISC_LOSS` and :data:`src.constants.OPTIMAL_GEN_LOSS`).
These functions are the single source of truth for the game's losses; the engine
calls them rather than inlining ``softplus`` arithmetic, so the math is tested in
isolation.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _check_logits(name: str, logits: Tensor) -> None:
    """Validate a 1-D, non-empty, floating-point logit vector at the boundary.

    Args:
        name: Argument name, used in error messages for attribution.
        logits: Candidate per-root logit tensor.

    Raises:
        TypeError: If ``logits`` is not a floating-point tensor.
        ValueError: If ``logits`` is not 1-D or is empty.
    """
    if not isinstance(logits, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(logits).__name__}.")
    if not torch.is_floating_point(logits):
        raise TypeError(f"{name} must have a floating dtype, got {logits.dtype}.")
    if logits.ndim != 1:
        raise ValueError(f"{name} must have shape (batch,), got shape {tuple(logits.shape)}.")
    if logits.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")


def discriminator_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    """Binary cross-entropy loss for the adaptive test function (eq. 4.1).

    Equivalent to ``-E[log D(real)] - E[log(1 - D(fake))]`` with
    ``D = sigmoid(logit)``. The two batches may differ in size (real and fake
    roots are sampled independently in general).

    Args:
        logits_real: Per-root logits on observed ego objects, shape ``(m_real,)``.
        logits_fake: Per-root logits on simulated ego objects, shape ``(m_fake,)``.

    Returns:
        Scalar loss tensor (carrying gradients to the discriminator). Equals
        ``2 log 2`` when all logits are zero (``D ≡ 1/2``).

    Raises:
        TypeError, ValueError: If either input violates the logit contract.
    """
    _check_logits("logits_real", logits_real)
    _check_logits("logits_fake", logits_fake)
    return F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()


def generator_nonsaturating_loss(logits_fake: Tensor) -> Tensor:
    """Non-saturating structural-parameter loss (eq. 4.2): ``-E[log D(fake)]``.

    This is the loss the structural parameters minimise. The non-saturating form
    keeps the gradient with respect to the outcome coordinates of order
    ``||grad_y log D||`` even as ``D(fake) -> 0``, avoiding the vanishing-gradient
    pathology of the saturating ``E[log(1 - D(fake))]`` form.

    Args:
        logits_fake: Per-root logits on simulated ego objects, shape ``(m,)``.

    Returns:
        Scalar loss tensor (carrying gradients through the discriminator to the
        structural parameters). Equals ``log 2`` when all logits are zero.

    Raises:
        TypeError, ValueError: If ``logits_fake`` violates the logit contract.
    """
    _check_logits("logits_fake", logits_fake)
    return F.softplus(-logits_fake).mean()


def generator_saturating_loss(logits_fake: Tensor) -> Tensor:
    """Saturating (original minimax) structural loss: ``E[log(1 - D(fake))]``.

    Provided for completeness and controlled comparison; the engine uses
    :func:`generator_nonsaturating_loss` by default. In logit form
    ``log(1 - D) = -softplus(logit)``, so this returns ``-softplus(logit_fake).mean()``.
    Its gradient vanishes as ``D(fake) -> 0``, which is exactly why the
    non-saturating variant is preferred for the structural update.

    Args:
        logits_fake: Per-root logits on simulated ego objects, shape ``(m,)``.

    Returns:
        Scalar loss tensor. Equals ``-log 2`` when all logits are zero.

    Raises:
        TypeError, ValueError: If ``logits_fake`` violates the logit contract.
    """
    _check_logits("logits_fake", logits_fake)
    return -F.softplus(logits_fake).mean()
