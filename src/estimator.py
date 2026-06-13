"""The adversarial estimation engine.

:class:`AdversarialEstimator` is the reusable, observable object form of the
alternating minimax estimator (Algorithm 1). It owns the structural model, the
adaptive test function (discriminator), the :class:`~src.ego.EgoSubstrate`, the
observed outcome vector, the optimisers, the stopping rule, and the observer fan-
out, and exposes a single :meth:`AdversarialEstimator.fit` entry point that
returns a typed :class:`~src.contracts.EstimationResult`.

It reproduces the training mechanics of the original Monte Carlo script exactly:

* paired focal nodes — real and simulated ego objects are compared at the *same*
  sampled roots within each discriminator step;
* one detached simulated equilibrium reused across the ``n_disc`` discriminator
  updates, and a fresh on-tape simulation for the structural update (independent
  shock draws for the two phases, per Section 4.1);
* the discriminator frozen (``requires_grad=False``) during the structural phase;
* the non-saturating structural loss (eq. 4.2), gradient clipping, and the
  optional instance-noise blur of Section 4.2;
* compounding learning-rate decay at the configured milestones.

The :attr:`AdversarialEstimator.gradient_transform` seam is where milestone 2's
Fisher / natural-gradient preconditioner plugs in: it runs after the structural
backward pass and before clipping, may mutate the model's ``.grad`` in place, and
may return diagnostics attached to that step's :class:`~src.contracts.StepMetrics`.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import utils as nn_utils

from .contracts import (
    EstimationResult,
    MetricsObserver,
    StepMetrics,
    StructuralModel,
    TestFunction,
)
from .ego import EgoSubstrate
from .estimator_config import EstimatorConfig
from .losses import (
    discriminator_loss,
    generator_nonsaturating_loss,
    generator_saturating_loss,
)
from .stopping import StoppingRule
from .utils import InstanceNoiseConfigLike, compute_instance_noise_taus

GradientTransform = Callable[["AdversarialEstimator", int], "Mapping[str, float] | None"]


class AdversarialEstimator:
    """Alternating-minimax adversarial structural estimator.

    Args:
        model: A :class:`~src.contracts.StructuralModel` (e.g. ``SCMGenerator`` or
            ``EffortGameGenerator``).
        discriminator: A :class:`~src.contracts.TestFunction` over rooted ego
            objects.
        substrate: The :class:`~src.ego.EgoSubstrate` (topology, ``W``, ``X``, ego
            cache, sampler) on which to estimate.
        Y_obs: Observed equilibrium outcome vector of shape ``(n,)``.
        config: Per-estimation :class:`~src.estimator_config.EstimatorConfig`.
        instance_noise: Optional instance-noise (blur) configuration applied to
            discriminator inputs; ``None`` disables it.
        observers: Observability sinks notified at run start, each step, and run
            end.
        gradient_transform: Optional structural-gradient transform applied after
            the structural backward pass and before clipping (milestone-2 seam).

    Raises:
        TypeError: If ``model``/``discriminator`` do not satisfy their protocols.
        ValueError: If ``Y_obs`` is the wrong shape.
    """

    def __init__(
        self,
        *,
        model: StructuralModel,
        discriminator: TestFunction,
        substrate: EgoSubstrate,
        Y_obs: Tensor,
        config: EstimatorConfig,
        instance_noise: InstanceNoiseConfigLike | None = None,
        observers: Sequence[MetricsObserver] = (),
        gradient_transform: GradientTransform | None = None,
    ) -> None:
        if not isinstance(model, StructuralModel):
            raise TypeError(
                "model must satisfy the StructuralModel protocol "
                "(callable (W, X) -> Y, get_params, parameters, named_parameters)."
            )
        if not isinstance(discriminator, TestFunction):
            raise TypeError(
                "discriminator must satisfy the TestFunction protocol "
                "(callable (x, edge_index, root_indices) -> logits)."
            )
        if not isinstance(substrate, EgoSubstrate):
            raise TypeError("substrate must be an EgoSubstrate instance.")
        if not isinstance(Y_obs, Tensor) or Y_obs.ndim != 1 or int(Y_obs.shape[0]) != substrate.num_nodes:
            raise ValueError(f"Y_obs must be a 1-D tensor of length {substrate.num_nodes}.")

        self.substrate = substrate
        self.device = substrate.device
        self.W = substrate.W
        self.X = substrate.X
        self.model = model.to(self.device)  # type: ignore[attr-defined]
        self.discriminator = discriminator.to(self.device)  # type: ignore[attr-defined]
        self.Y_obs = Y_obs.detach().to(self.device)
        self.config = config
        self.instance_noise = instance_noise
        self.observers = list(observers)
        self.gradient_transform = gradient_transform

        self.norm_stats = substrate.make_norm_stats(self.Y_obs)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr_d)
        self.opt_g = torch.optim.Adam(self.model.parameters(), lr=config.lr_g)
        self._stopping = StoppingRule(config)
        self._decay_steps = set(int(s) for s in config.lr_g_decay_steps)

    # ----------------------------------------------------------------- internals
    def _simulate(self, *, detached: bool) -> Tensor:
        """Simulate one equilibrium outcome with fresh shocks.

        Args:
            detached: If true, simulate under ``no_grad`` (discriminator phase);
                otherwise keep the unrolled solve on the autograd tape (structural
                phase).
        """
        if detached:
            with torch.no_grad():
                return self.model(self.W, self.X)
        return self.model(self.W, self.X)

    def _logits(self, Y: Tensor, roots: Tensor, *, step: int, role: str) -> Tensor:
        """Build a rooted-ego batch for ``Y`` at ``roots`` and return per-root logits."""
        batch, root_indices = self.substrate.build_batch(
            roots, Y, self.norm_stats, step=step, role=role, instance_noise=self.instance_noise
        )
        return self.discriminator(batch.x, batch.edge_index, root_indices)

    def _set_discriminator_grad(self, requires_grad: bool) -> None:
        for param in self.discriminator.parameters():
            param.requires_grad_(requires_grad)

    def _emit(self, method: str, payload: object) -> None:
        """Dispatch an observability event, isolating observer failures."""
        for observer in self.observers:
            try:
                getattr(observer, method)(payload)
            except Exception as exc:  # pragma: no cover - defensive observability guard
                warnings.warn(
                    f"observer {type(observer).__name__}.{method} raised: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _failure(
        self,
        *,
        step: int,
        reason: str,
        param_history: Mapping[str, list[float]],
        loss_d_rolling: float,
        loss_g_rolling: float,
    ) -> EstimationResult:
        last = {name: (path[-1] if path else float("nan")) for name, path in param_history.items()}
        result = EstimationResult(
            status=f"failed:{reason}",
            converged=False,
            final_step=int(step),
            params={name: float("nan") for name in param_history},
            params_final=last,
            loss_d_rolling_final=loss_d_rolling,
            loss_g_rolling_final=loss_g_rolling,
            n_steps_run=int(step),
            failure_reason=reason,
        )
        self._emit("on_run_end", result)
        return result

    # ---------------------------------------------------------------------- fit
    def fit(self) -> EstimationResult:
        """Run the alternating minimax estimation to its stopping rule.

        Returns:
            A typed :class:`~src.contracts.EstimationResult` with the tail-averaged
            point estimate, the final iterate, and convergence diagnostics. On a
            structural failure (non-finite simulation or NaN loss) the result has a
            ``"failed:<reason>"`` status and the partial history is preserved in any
            in-memory observer.
        """
        cfg = self.config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            self.substrate.root_sampler.rng = np.random.default_rng(cfg.seed)

        self._emit(
            "on_run_start",
            {
                "model": type(self.model).__name__,
                "discriminator": type(self.discriminator).__name__,
                "num_nodes": self.substrate.num_nodes,
                "k": self.substrate.k,
                "params": list(self.model.get_params().keys()),
                "config": cfg,
            },
        )

        loss_d_history: list[float] = []
        loss_g_history: list[float] = []
        param_history: dict[str, list[float]] = {name: [] for name in self.model.get_params()}

        converged = False
        final_step = 0
        decision_d_rolling = float("nan")
        decision_g_rolling = float("nan")

        for step in range(1, cfg.max_steps + 1):
            final_step = step
            if step in self._decay_steps:
                for group in self.opt_g.param_groups:
                    group["lr"] *= cfg.lr_g_decay_factor
            lr_g = float(self.opt_g.param_groups[0]["lr"])

            self.discriminator.train()

            # ---- Discriminator phase: one detached simulation reused across n_disc.
            Y_sim_detached = self._simulate(detached=True)
            if not torch.isfinite(Y_sim_detached).all():
                return self._failure(
                    step=step, reason="Y_sim_non_finite_D", param_history=param_history,
                    loss_d_rolling=decision_d_rolling, loss_g_rolling=decision_g_rolling,
                )

            last_loss_d = float("nan")
            self._set_discriminator_grad(True)
            for _ in range(cfg.n_disc):
                roots, _ = self.substrate.sample_roots(cfg.batch_size)
                logits_real = self._logits(self.Y_obs, roots, step=step, role="real")
                logits_fake = self._logits(Y_sim_detached, roots, step=step, role="fake")
                self.opt_d.zero_grad(set_to_none=True)
                loss_d = discriminator_loss(logits_real, logits_fake)
                loss_d.backward()
                self.opt_d.step()
                last_loss_d = float(loss_d.item())

            # ---- Structural phase: fresh on-tape simulation, discriminator frozen.
            self._set_discriminator_grad(False)
            roots_g, root_result = self.substrate.sample_roots(cfg.batch_size)
            Y_sim = self._simulate(detached=False)
            if not torch.isfinite(Y_sim).all():
                return self._failure(
                    step=step, reason="Y_sim_non_finite_G", param_history=param_history,
                    loss_d_rolling=decision_d_rolling, loss_g_rolling=decision_g_rolling,
                )

            logits_fake_g = self._logits(Y_sim, roots_g, step=step, role="fake")
            self.opt_g.zero_grad(set_to_none=True)
            if cfg.nonsaturating:
                loss_g = generator_nonsaturating_loss(logits_fake_g)
            else:
                loss_g = generator_saturating_loss(logits_fake_g)
            loss_g.backward()

            extras: dict[str, float] = {}
            if self.gradient_transform is not None:
                transform_extras = self.gradient_transform(self, step)
                if transform_extras:
                    extras.update(transform_extras)

            grad_norm_g = _total_grad_norm(self.model.parameters())
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip_norm)
            self.opt_g.step()

            # ---- Bookkeeping, diagnostics, stopping.
            params = self.model.get_params()
            loss_g_value = float(loss_g.item())
            tau_x, tau_y = compute_instance_noise_taus(self.instance_noise, generator_step=step)

            loss_d_history.append(last_loss_d)
            loss_g_history.append(loss_g_value)
            for name, value in params.items():
                param_history[name].append(float(value))

            if math.isnan(last_loss_d) or math.isnan(loss_g_value):
                return self._failure(
                    step=step, reason="nan_loss", param_history=param_history,
                    loss_d_rolling=decision_d_rolling, loss_g_rolling=decision_g_rolling,
                )

            decision = self._stopping.evaluate(loss_d_history, loss_g_history, param_history)
            decision_d_rolling = decision.loss_d_rolling
            decision_g_rolling = decision.loss_g_rolling

            self._emit(
                "on_step",
                StepMetrics(
                    step=step,
                    params=params,
                    loss_d=last_loss_d,
                    loss_g=loss_g_value,
                    loss_d_rolling=decision.loss_d_rolling,
                    loss_g_rolling=decision.loss_g_rolling,
                    grad_norm_g=grad_norm_g,
                    picard_iterations=int(getattr(self.model, "last_picard_iterations", 0)),
                    roots_requested=int(root_result.requested_size),
                    roots_achieved=int(root_result.achieved_size),
                    tau_x=float(tau_x),
                    tau_y=float(tau_y),
                    in_equilibrium=decision.in_equilibrium,
                    lr_g=lr_g,
                    newton_iterations=_maybe_int(getattr(self.model, "last_newton_max_iters", None)),
                    sampler_radius=root_result.radius_used,
                    extras=extras,
                ),
            )

            if decision.converged:
                converged = True
                break

        result = self._finalize(
            converged=converged,
            final_step=final_step,
            param_history=param_history,
            loss_d_history=loss_d_history,
            loss_g_history=loss_g_history,
        )
        self._emit("on_run_end", result)
        return result

    def _finalize(
        self,
        *,
        converged: bool,
        final_step: int,
        param_history: Mapping[str, list[float]],
        loss_d_history: Sequence[float],
        loss_g_history: Sequence[float],
    ) -> EstimationResult:
        """Assemble the tail-averaged estimate and final diagnostics."""
        cfg = self.config
        n_logged = len(loss_d_history)
        tail_window = min(max(cfg.convergence_window, cfg.stability_window), n_logged) if n_logged else 0

        point_estimate: dict[str, float] = {}
        final_params: dict[str, float] = {}
        for name, path in param_history.items():
            if path:
                final_params[name] = float(path[-1])
                point_estimate[name] = float(np.mean(path[-tail_window:])) if tail_window else float(path[-1])
            else:
                final_params[name] = float("nan")
                point_estimate[name] = float("nan")

        loss_window = min(cfg.convergence_window, n_logged)
        if loss_window > 0:
            loss_d_rolling_final = float(np.mean(loss_d_history[-loss_window:]))
            loss_g_rolling_final = float(np.mean(loss_g_history[-loss_window:]))
        else:
            loss_d_rolling_final = float("nan")
            loss_g_rolling_final = float("nan")

        return EstimationResult(
            status="ok",
            converged=converged,
            final_step=int(final_step),
            params=point_estimate,
            params_final=final_params,
            loss_d_rolling_final=loss_d_rolling_final,
            loss_g_rolling_final=loss_g_rolling_final,
            n_steps_run=n_logged,
        )


def _total_grad_norm(params) -> float:
    """L2 norm of the concatenated gradients of ``params`` (skipping ``None``)."""
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _maybe_int(value: object) -> int | None:
    """Coerce an optional iteration counter to ``int`` or ``None``."""
    if value is None:
        return None
    return int(value)  # type: ignore[arg-type]
