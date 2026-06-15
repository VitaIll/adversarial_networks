"""Stopping rule for the adversarial estimation engine.

The estimator stops when two model-agnostic conditions hold simultaneously over a
trailing window:

1. **Loss-band convergence** — the rolling-mean discriminator and structural
   losses sit within tolerance of their theoretical equilibrium values at
   ``D* ≡ 1/2`` (discriminator ``2 log 2``; generator ``log 2`` for the
   non-saturating loss or ``-log 2`` for the saturating loss), with bounded
   rolling std. This reuses the tested
   :func:`~adversarial_networks.core.objective.check_gan_convergence`.
2. **Parameter stabilisation** — every estimated structural parameter has a
   trailing-window range (max - min) within its tolerance. Unlike the original
   script (which hard-codes ``beta``/``gamma``/``sigma_sq``), this generalises to
   the parameter set returned by any model's ``get_params``.

The rule is the operational form of the paper's diagnostic (Section 5.2): at a
local minimiser of the empirical criterion both losses sit at the equilibrium
benchmark and the structural path is flat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .core.objective import (
    OPTIMAL_GEN_LOSS,
    OPTIMAL_GEN_LOSS_SATURATING,
    check_gan_convergence,
)
from .estimator_config import EstimatorConfig


@dataclass(frozen=True)
class StoppingDecision:
    """Outcome of one stopping-rule evaluation.

    Attributes:
        converged: Whether both the loss band and parameter stability hold and the
            minimum step count is reached.
        in_equilibrium: Whether the loss-band check alone passes this step (used as
            an observability flag even before parameters stabilise).
        loss_d_rolling: Rolling-mean discriminator loss (``nan`` until the window
            fills).
        loss_g_rolling: Rolling-mean structural loss (``nan`` until filled).
        params_stable: Whether the parameter-stability condition holds.
    """

    converged: bool
    in_equilibrium: bool
    loss_d_rolling: float
    loss_g_rolling: float
    params_stable: bool


class StoppingRule:
    """Model-agnostic loss-band + parameter-stability stopping rule."""

    def __init__(self, config: EstimatorConfig) -> None:
        self._config = config

    def _params_stable(self, param_history: Mapping[str, Sequence[float]]) -> bool:
        """Whether every parameter path is flat over the trailing stability window.

        Returns ``False`` if any tracked path is shorter than the window, so the
        rule cannot fire before enough history accrues.
        """
        window = self._config.stability_window
        if not param_history:
            return False
        for name, path in param_history.items():
            if len(path) < window:
                return False
            tail = path[-window:]
            value_range = max(tail) - min(tail)
            if value_range > self._config.override_tol_for(name):
                return False
        return True

    def evaluate(
        self,
        loss_d_history: Sequence[float],
        loss_g_history: Sequence[float],
        param_history: Mapping[str, Sequence[float]],
    ) -> StoppingDecision:
        """Evaluate the stopping rule given the current histories.

        Args:
            loss_d_history: Per-step discriminator losses.
            loss_g_history: Per-step structural losses.
            param_history: Per-parameter value paths keyed by parameter name.

        Returns:
            A :class:`StoppingDecision`.
        """
        in_equilibrium, rolling_d, rolling_g = check_gan_convergence(
            loss_d_history=list(loss_d_history),
            loss_g_history=list(loss_g_history),
            window=self._config.convergence_window,
            delta_d=self._config.convergence_delta_d,
            delta_g=self._config.convergence_delta_g,
            min_steps=self._config.min_steps,
            std_d_max=self._config.convergence_std_d_max,
            std_g_max=self._config.convergence_std_g_max,
            gen_optimum=(
                OPTIMAL_GEN_LOSS
                if self._config.nonsaturating
                else OPTIMAL_GEN_LOSS_SATURATING
            ),
        )
        params_stable = self._params_stable(param_history)
        return StoppingDecision(
            converged=bool(in_equilibrium and params_stable),
            in_equilibrium=bool(in_equilibrium),
            loss_d_rolling=rolling_d,
            loss_g_rolling=rolling_g,
            params_stable=params_stable,
        )
