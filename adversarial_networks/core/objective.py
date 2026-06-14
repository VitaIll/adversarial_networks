"""The adversarial criterion: minimax losses, convergence math, blur schedule.

This is the single source of truth for the game's objective. It is pure and
dependency-light (``torch`` + stdlib ``math`` only — no ``torch_geometric``), so
the criterion math is unit-testable in isolation.

* :func:`discriminator_loss` / :func:`generator_nonsaturating_loss` /
  :func:`generator_saturating_loss` — the two minimax losses (Algorithm 1,
  eq. 4.1 / 4.2). At the population optimum ``D* ≡ 1/2`` they sit at
  :data:`OPTIMAL_DISC_LOSS` ``= 2 log 2`` and :data:`OPTIMAL_GEN_LOSS` ``= log 2``.
* :func:`check_gan_convergence` — the loss-band diagnostic (Section 5.2).
* :func:`instance_noise_taus` — the annealed instance-noise (blur) schedule of
  Section 4.2.

References:
    Illichmann & Zacchia (2026), Algorithm 1, Sections 4.2 and 5.2.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .types import InstanceNoiseConfigLike

OPTIMAL_DISC_LOSS = 2.0 * math.log(2.0)
"""Theoretical discriminator loss at the population optimum (``D ≡ 1/2``) ≈ 1.386."""

OPTIMAL_GEN_LOSS = math.log(2.0)
"""Theoretical (non-saturating) generator loss at the population optimum ≈ 0.693."""


# --------------------------------------------------------------------- losses
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

    The non-saturating form keeps the gradient with respect to the outcome
    coordinates of order ``||grad_y log D||`` even as ``D(fake) -> 0``, avoiding
    the vanishing-gradient pathology of the saturating ``E[log(1 - D(fake))]`` form.

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

    Provided for completeness and controlled comparison; the estimator uses
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


# ---------------------------------------------------------------- convergence
def check_gan_convergence(
    loss_d_history: list[float],
    loss_g_history: list[float],
    window: int,
    delta_d: float,
    delta_g: float,
    min_steps: int | None = None,
    std_d_max: float | None = None,
    std_g_max: float | None = None,
) -> tuple[bool, float, float]:
    """Check whether the minimax losses have stabilised near equilibrium.

    Args:
        loss_d_history: Per-step discriminator loss history.
        loss_g_history: Per-step generator loss history.
        window: Rolling mean window size.
        delta_d: Absolute tolerance around :data:`OPTIMAL_DISC_LOSS` (``2 log 2``).
        delta_g: Absolute tolerance around :data:`OPTIMAL_GEN_LOSS` (``log 2``).
        min_steps: Optional earliest step where convergence can be declared.
        std_d_max: Optional upper bound for the rolling discriminator-loss std.
        std_g_max: Optional upper bound for the rolling generator-loss std.

    Returns:
        Tuple ``(converged, rolling_mean_d, rolling_mean_g)``; the rolling means
        are ``nan`` until enough observations exist for the configured window and
        minimum step requirement.

    Raises:
        ValueError: If history lengths differ or thresholds are invalid.
    """
    if len(loss_d_history) != len(loss_g_history):
        raise ValueError(
            "loss_d_history and loss_g_history must have equal length, got "
            f"{len(loss_d_history)} and {len(loss_g_history)}"
        )
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if delta_d <= 0.0:
        raise ValueError(f"delta_d must be positive, got {delta_d}")
    if delta_g <= 0.0:
        raise ValueError(f"delta_g must be positive, got {delta_g}")
    if min_steps is not None and min_steps < 0:
        raise ValueError(f"min_steps must be non-negative when provided, got {min_steps}")
    if std_d_max is not None and std_d_max <= 0.0:
        raise ValueError(f"std_d_max must be positive when provided, got {std_d_max}")
    if std_g_max is not None and std_g_max <= 0.0:
        raise ValueError(f"std_g_max must be positive when provided, got {std_g_max}")

    effective_min_steps = int(min_steps) if min_steps is not None else 0
    step = len(loss_d_history)
    if step < effective_min_steps or step < window:
        return False, float("nan"), float("nan")

    tail_d = loss_d_history[-window:]
    tail_g = loss_g_history[-window:]
    rolling_d = sum(tail_d) / float(window)
    rolling_g = sum(tail_g) / float(window)
    var_d = sum((value - rolling_d) ** 2 for value in tail_d) / float(window)
    var_g = sum((value - rolling_g) ** 2 for value in tail_g) / float(window)
    std_d = math.sqrt(max(0.0, var_d))
    std_g = math.sqrt(max(0.0, var_g))

    d_ok = abs(rolling_d - OPTIMAL_DISC_LOSS) < delta_d
    g_ok = abs(rolling_g - OPTIMAL_GEN_LOSS) < delta_g
    if std_d_max is not None:
        d_ok = d_ok and (std_d <= std_d_max)
    if std_g_max is not None:
        g_ok = g_ok and (std_g <= std_g_max)
    return (d_ok and g_ok), rolling_d, rolling_g


# -------------------------------------------------------------- instance noise
def instance_noise_taus(
    instance_noise: InstanceNoiseConfigLike | None,
    generator_step: int,
) -> tuple[float, float]:
    """Compute the scheduled blur intensity (normalised units) for a step.

    Args:
        instance_noise: Instance-noise configuration. If ``None`` or disabled,
            returns ``(0.0, 0.0)``.
        generator_step: Outer generator step counter (0- or 1-based).

    Returns:
        ``(tau_x, tau_y)`` in normalised units.
    """
    if instance_noise is None or not bool(instance_noise.enabled):
        return 0.0, 0.0

    step = max(0, int(generator_step))
    schedule = str(instance_noise.schedule)
    anneal_steps = int(instance_noise.anneal_steps)
    min_tau = float(instance_noise.min_tau)

    if schedule not in {"constant", "linear", "exp"}:
        raise ValueError(
            "instance-noise schedule must be one of {'constant', 'linear', 'exp'}, "
            f"got {schedule!r}."
        )
    if anneal_steps < 0:
        raise ValueError(f"anneal_steps must be non-negative, got {anneal_steps}.")
    if min_tau < 0.0:
        raise ValueError(f"min_tau must be non-negative, got {min_tau}.")

    def _tau_at_step(tau0: float) -> float:
        tau0 = float(tau0)
        if tau0 < 0.0:
            raise ValueError(f"tau0 must be non-negative, got {tau0}.")
        if schedule == "constant" or anneal_steps == 0:
            return max(min_tau, tau0)
        if schedule == "linear":
            if step <= anneal_steps:
                frac = 1.0 - (float(step) / float(anneal_steps))
                return max(min_tau, tau0 * frac)
            return max(min_tau, min_tau)
        # Exponential annealing with decay derived from anneal_steps.
        tau_decay = max(1.0, float(anneal_steps) / 5.0)
        return max(min_tau, tau0 * math.exp(-float(step) / tau_decay))

    tau_x = _tau_at_step(float(instance_noise.tau_x0))
    tau_y = _tau_at_step(float(instance_noise.tau_y0))
    return tau_x, tau_y
