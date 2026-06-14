"""The adversarial estimation engine (sklearn / DoubleML-shaped).

:class:`AdversarialEstimator` is a single, model-agnostic estimator object. Its
``__init__`` stores arguments verbatim (no work — the sklearn contract); ``fit``
does the work and returns ``self``, exposing trailing-underscore learned
attributes (``model_``, ``params_``, ``estimates_``, …). Accessing a learned
attribute before ``fit`` raises :class:`NotFittedError`.

The verified training mechanics live in the free function :func:`_run_minimax`,
called by both ``fit`` and :class:`~adversarial_networks.runner.MonteCarloRunner`
(a frozen-mechanics seam, not a parallel class). It reproduces the original Monte
Carlo loop exactly:

* paired focal nodes — real and simulated ego objects compared at the *same* roots;
* one detached simulated equilibrium reused across the ``n_disc`` discriminator
  updates, and a fresh on-tape simulation for the structural update;
* the discriminator frozen during the structural phase;
* the non-saturating structural loss (eq. 4.2), instance-noise blur, gradient
  clipping, and compounding learning-rate decay.

:class:`MinimaxStepContext` is the documented entry point for the future
true-Fisher / GGN preconditioner (the ``gradient_transform`` seam): a frozen
snapshot of the estimation state, passed to the transform after the structural
backward pass and before gradient clipping.

References:
    Illichmann & Zacchia (2026), Algorithm 1 and Section 4.2.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

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
from .core.objective import instance_noise_taus as compute_instance_noise_taus
from .core.types import InstanceNoiseConfigLike
from .data import NetworkData
from .ego import EgoSubstrate
from .estimator_config import EstimatorConfig
from .losses import (
    discriminator_loss,
    generator_nonsaturating_loss,
    generator_saturating_loss,
)
from .observability import InMemoryHistory
from .stopping import StoppingRule


class NotFittedError(ValueError, AttributeError):
    """Raised when a learned attribute is accessed before ``fit`` (sklearn idiom)."""


class EstimationFailedWarning(UserWarning):
    """Emitted when a structural failure (non-finite simulation / NaN loss) aborts a fit."""


class ConvergenceWarning(UserWarning):
    """Emitted when the stopping rule did not fire before ``max_steps``."""


@dataclass(frozen=True)
class MinimaxStepContext:
    """Frozen snapshot of the estimation state at one structural step.

    Passed to a ``gradient_transform`` after the structural backward pass and before
    clipping. Carries everything the future Fisher / GGN preconditioner needs, so
    the seam will not force a later signature change.
    """

    step: int
    model: StructuralModel
    discriminator: TestFunction
    substrate: EgoSubstrate
    config: EstimatorConfig
    instance_noise: InstanceNoiseConfigLike | None
    Y_obs: Tensor
    norm_stats: Mapping[str, float]

    @property
    def W(self) -> Tensor:
        return self.substrate.W

    @property
    def X(self) -> Tensor:
        return self.substrate.X


GradientTransform = Callable[[MinimaxStepContext], "Mapping[str, float] | None"]


# ====================================================================== engine
def _run_minimax(
    *,
    model: StructuralModel,
    discriminator: TestFunction,
    substrate: EgoSubstrate,
    Y_obs: Tensor,
    config: EstimatorConfig,
    instance_noise: InstanceNoiseConfigLike | None = None,
    observers: Sequence[MetricsObserver] = (),
    gradient_transform: GradientTransform | None = None,
    device: torch.device | None = None,
) -> EstimationResult:
    """Run the alternating-minimax estimation to its stopping rule (the frozen loop).

    Operates on the *given* model/discriminator in place (the caller owns cloning):
    ``fit`` passes deep copies; the Monte Carlo runner passes fresh per-realisation
    objects. Returns a typed :class:`~adversarial_networks.contracts.EstimationResult`.
    """
    device = device or substrate.device
    model = model.to(device)  # type: ignore[attr-defined]
    discriminator = discriminator.to(device)  # type: ignore[attr-defined]
    Y_obs = Y_obs.detach().to(device)

    layers = getattr(discriminator, "num_layers", None)
    if layers is not None and int(layers) < substrate.k:
        raise ValueError(
            f"discriminator num_layers ({layers}) must be >= the ego radius k "
            f"({substrate.k}) so the test function covers the ego neighbourhood "
            "(the paper's '>= k message-passing layers')."
        )

    norm_stats = substrate.make_norm_stats(Y_obs)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d)
    opt_g = torch.optim.Adam(model.parameters(), lr=config.lr_g)
    stopping = StoppingRule(config)
    decay_steps = {int(s) for s in config.lr_g_decay_steps}
    observers = list(observers)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        substrate.root_sampler.rng = np.random.default_rng(config.seed)

    def emit(method: str, payload: object) -> None:
        for observer in observers:
            try:
                getattr(observer, method)(payload)
            except Exception as exc:  # pragma: no cover - defensive observability guard
                warnings.warn(
                    f"observer {type(observer).__name__}.{method} raised: {exc}",
                    RuntimeWarning, stacklevel=2,
                )

    def simulate(detached: bool) -> Tensor:
        if detached:
            with torch.no_grad():
                return model(substrate.W, substrate.X)
        return model(substrate.W, substrate.X)

    def logits(Y: Tensor, roots: Tensor, *, step: int, role: str) -> Tensor:
        batch, root_indices = substrate.build_batch(
            roots, Y, norm_stats, step=step, role=role, instance_noise=instance_noise
        )
        return discriminator(batch.x, batch.edge_index, root_indices)

    def set_disc_grad(flag: bool) -> None:
        for param in discriminator.parameters():
            param.requires_grad_(flag)

    emit("on_run_start", {
        "model": type(model).__name__,
        "discriminator": type(discriminator).__name__,
        "num_nodes": substrate.num_nodes,
        "k": substrate.k,
        "params": list(model.get_params().keys()),
        "config": config,
    })

    loss_d_history: list[float] = []
    loss_g_history: list[float] = []
    param_history: dict[str, list[float]] = {name: [] for name in model.get_params()}
    converged = False
    final_step = 0
    decision_d_rolling = float("nan")
    decision_g_rolling = float("nan")

    def failure(step: int, reason: str) -> EstimationResult:
        last = {name: (path[-1] if path else float("nan")) for name, path in param_history.items()}
        result = EstimationResult(
            status=f"failed:{reason}", converged=False, final_step=int(step),
            params={name: float("nan") for name in param_history}, params_final=last,
            loss_d_rolling_final=decision_d_rolling, loss_g_rolling_final=decision_g_rolling,
            n_steps_run=int(step), failure_reason=reason,
        )
        emit("on_run_end", result)
        return result

    for step in range(1, config.max_steps + 1):
        final_step = step
        if step in decay_steps:
            for group in opt_g.param_groups:
                group["lr"] *= config.lr_g_decay_factor
        lr_g = float(opt_g.param_groups[0]["lr"])

        discriminator.train()

        # ---- Discriminator phase: one detached simulation reused across n_disc.
        Y_sim_detached = simulate(detached=True)
        if not torch.isfinite(Y_sim_detached).all():
            return failure(step, "Y_sim_non_finite_D")

        last_loss_d = float("nan")
        set_disc_grad(True)
        for _ in range(config.n_disc):
            roots, _ = substrate.sample_roots(config.batch_size)
            logits_real = logits(Y_obs, roots, step=step, role="real")
            logits_fake = logits(Y_sim_detached, roots, step=step, role="fake")
            opt_d.zero_grad(set_to_none=True)
            loss_d = discriminator_loss(logits_real, logits_fake)
            loss_d.backward()
            opt_d.step()
            last_loss_d = float(loss_d.item())

        # ---- Structural phase: fresh on-tape simulation, discriminator frozen.
        set_disc_grad(False)
        roots_g, root_result = substrate.sample_roots(config.batch_size)
        Y_sim = simulate(detached=False)
        if not torch.isfinite(Y_sim).all():
            return failure(step, "Y_sim_non_finite_G")

        logits_fake_g = logits(Y_sim, roots_g, step=step, role="fake")
        opt_g.zero_grad(set_to_none=True)
        loss_g = (
            generator_nonsaturating_loss(logits_fake_g)
            if config.nonsaturating
            else generator_saturating_loss(logits_fake_g)
        )
        loss_g.backward()

        extras: dict[str, float] = {}
        if gradient_transform is not None:
            context = MinimaxStepContext(
                step=step, model=model, discriminator=discriminator, substrate=substrate,
                config=config, instance_noise=instance_noise, Y_obs=Y_obs, norm_stats=norm_stats,
            )
            transform_extras = gradient_transform(context)
            if transform_extras:
                extras.update(transform_extras)
            # Anomaly localisation: the transform must leave finite, shape-correct grads.
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if param.grad.shape != param.shape:
                        raise RuntimeError(
                            f"gradient_transform changed grad shape of {name!r}: "
                            f"{tuple(param.grad.shape)} != {tuple(param.shape)}."
                        )
                    if not bool(torch.isfinite(param.grad).all()):
                        raise RuntimeError(
                            f"gradient_transform produced a non-finite grad for {name!r}."
                        )

        grad_norm_g = _total_grad_norm(model.parameters())
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        opt_g.step()

        # ---- Bookkeeping, diagnostics, stopping.
        params = model.get_params()
        loss_g_value = float(loss_g.item())
        tau_x, tau_y = compute_instance_noise_taus(instance_noise, generator_step=step)

        loss_d_history.append(last_loss_d)
        loss_g_history.append(loss_g_value)
        for name, value in params.items():
            param_history[name].append(float(value))

        if math.isnan(last_loss_d) or math.isnan(loss_g_value):
            return failure(step, "nan_loss")

        decision = stopping.evaluate(loss_d_history, loss_g_history, param_history)
        decision_d_rolling = decision.loss_d_rolling
        decision_g_rolling = decision.loss_g_rolling

        emit("on_step", StepMetrics(
            step=step, params=params, loss_d=last_loss_d, loss_g=loss_g_value,
            loss_d_rolling=decision.loss_d_rolling, loss_g_rolling=decision.loss_g_rolling,
            grad_norm_g=grad_norm_g,
            picard_iterations=int(getattr(model, "last_picard_iterations", 0)),
            roots_requested=int(root_result.requested_size),
            roots_achieved=int(root_result.achieved_size),
            tau_x=float(tau_x), tau_y=float(tau_y), in_equilibrium=decision.in_equilibrium,
            lr_g=lr_g, newton_iterations=_maybe_int(getattr(model, "last_newton_max_iters", None)),
            sampler_radius=root_result.radius_used, extras=extras,
        ))

        if decision.converged:
            converged = True
            break

    result = _finalize(
        converged=converged, final_step=final_step, config=config,
        param_history=param_history, loss_d_history=loss_d_history, loss_g_history=loss_g_history,
    )
    emit("on_run_end", result)
    return result


def _finalize(
    *,
    converged: bool,
    final_step: int,
    config: EstimatorConfig,
    param_history: Mapping[str, list[float]],
    loss_d_history: Sequence[float],
    loss_g_history: Sequence[float],
) -> EstimationResult:
    """Assemble the tail-averaged estimate and final diagnostics."""
    n_logged = len(loss_d_history)
    tail_window = (
        min(max(config.convergence_window, config.stability_window), n_logged) if n_logged else 0
    )

    point_estimate: dict[str, float] = {}
    final_params: dict[str, float] = {}
    for name, path in param_history.items():
        if path:
            final_params[name] = float(path[-1])
            point_estimate[name] = (
                float(np.mean(path[-tail_window:])) if tail_window else float(path[-1])
            )
        else:
            final_params[name] = float("nan")
            point_estimate[name] = float("nan")

    loss_window = min(config.convergence_window, n_logged)
    if loss_window > 0:
        loss_d_rolling_final = float(np.mean(loss_d_history[-loss_window:]))
        loss_g_rolling_final = float(np.mean(loss_g_history[-loss_window:]))
    else:
        loss_d_rolling_final = float("nan")
        loss_g_rolling_final = float("nan")

    return EstimationResult(
        status="ok", converged=converged, final_step=int(final_step),
        params=point_estimate, params_final=final_params,
        loss_d_rolling_final=loss_d_rolling_final, loss_g_rolling_final=loss_g_rolling_final,
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
    if value is None:
        return None
    return int(value)  # type: ignore[arg-type]


# ============================================================= public estimator
class AdversarialEstimator:
    """Alternating-minimax adversarial structural estimator (sklearn/DoubleML-shaped).

    Args:
        model: A :class:`~adversarial_networks.contracts.StructuralModel` (a
            ``NetworkGameGenerator`` instance or anything satisfying the protocol).
        discriminator: A :class:`~adversarial_networks.contracts.TestFunction`.
        config: The :class:`~adversarial_networks.estimator_config.EstimatorConfig`
            (defaults to ``EstimatorConfig.recovery_default()`` when ``None``).
        max_steps, batch_size, lr_d, lr_g, seed: Headline shortcut overrides — a
            non-``None`` value replaces the corresponding config field at ``fit``.
        instance_noise: Optional discriminator-input blur configuration.
        gradient_transform: Optional structural-gradient transform (the future-Fisher
            seam), called with a :class:`MinimaxStepContext` after the structural
            backward and before clipping.
        observers: Observability sinks (an :class:`InMemoryHistory` is always added).
        device: Optional device override (defaults to the data's device).
    """

    # The ``estimates_`` property raises NotFittedError (an AttributeError subclass)
    # before fit, which falls through to __getattr__ — listing it here makes that
    # fallback re-raise NotFittedError rather than a bare AttributeError.
    _LEARNED = frozenset({
        "model_", "discriminator_", "result_", "history_", "params_", "params_final_",
        "converged_", "n_iter_", "loss_d_", "loss_g_", "feature_names_", "estimates_",
    })

    def __init__(
        self,
        model: StructuralModel,
        discriminator: TestFunction,
        *,
        config: EstimatorConfig | None = None,
        max_steps: int | None = None,
        batch_size: int | None = None,
        lr_d: float | None = None,
        lr_g: float | None = None,
        seed: int | None = None,
        instance_noise: InstanceNoiseConfigLike | None = None,
        gradient_transform: GradientTransform | None = None,
        observers: Sequence[MetricsObserver] = (),
        device: torch.device | str | None = None,
    ) -> None:
        # Store verbatim — no logic in __init__ (sklearn contract / clone-safety).
        self.model = model
        self.discriminator = discriminator
        self.config = config
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.seed = seed
        self.instance_noise = instance_noise
        self.gradient_transform = gradient_transform
        self.observers = observers
        self.device = device

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails — i.e. a learned attr before fit.
        if name in type(self)._LEARNED:
            raise NotFittedError(
                "This AdversarialEstimator is not fitted yet; call fit(data) before "
                f"accessing {name!r}."
            )
        raise AttributeError(name)

    def _check_fitted(self) -> None:
        if "result_" not in self.__dict__:
            raise NotFittedError("This AdversarialEstimator is not fitted yet; call fit(data) first.")

    def fit(self, data: NetworkData) -> AdversarialEstimator:
        """Fit on a :class:`~adversarial_networks.data.NetworkData` and return ``self``.

        Deep-copies the model/discriminator (clone-safe; the ctor objects stay
        pristine), applies the shortcut overrides + ``config.differentiation``, runs
        :func:`_run_minimax`, stores the learned attributes, and warns on a
        structural failure or non-convergence.
        """
        if not isinstance(data, NetworkData):
            raise TypeError(f"data must be a NetworkData, got {type(data).__name__}.")

        config = self.config if self.config is not None else EstimatorConfig.recovery_default()
        overrides = {
            field: value
            for field, value in (
                ("max_steps", self.max_steps), ("batch_size", self.batch_size),
                ("lr_d", self.lr_d), ("lr_g", self.lr_g), ("seed", self.seed),
            )
            if value is not None
        }
        if overrides:
            config = replace(config, **overrides)

        layers = getattr(self.discriminator, "num_layers", None)
        if layers is not None and int(layers) < data.k:
            raise ValueError(
                f"discriminator num_layers ({layers}) must be >= data.k ({data.k}); the "
                "test function must cover the ego radius (>= k message-passing layers)."
            )

        # Clone-safety: fit must not mutate the caller's modules or warm-start.
        self.model_ = copy.deepcopy(self.model)
        self.discriminator_ = copy.deepcopy(self.discriminator)
        if config.differentiation is not None and hasattr(self.model_, "differentiation"):
            self.model_.differentiation = config.differentiation

        history = InMemoryHistory()
        observers = list(self.observers) + [history]

        result = _run_minimax(
            model=self.model_, discriminator=self.discriminator_, substrate=data.topology,
            Y_obs=data.y, config=config, instance_noise=self.instance_noise,
            observers=observers, gradient_transform=self.gradient_transform, device=self.device,
        )

        self.result_ = result
        self.history_ = history
        self.params_ = dict(result.params)
        self.params_final_ = dict(result.params_final)
        self.converged_ = bool(result.converged)
        self.n_iter_ = int(result.n_steps_run)
        self.loss_d_ = float(result.loss_d_rolling_final)
        self.loss_g_ = float(result.loss_g_rolling_final)
        self.feature_names_ = list(result.params.keys())
        self._train_substrate = data.topology
        self._train_y = data.y
        self._config_used = config

        if not result.ok:
            warnings.warn(
                f"estimation failed: {result.failure_reason}; params_ are NaN.",
                EstimationFailedWarning, stacklevel=2,
            )
        elif not result.converged:
            warnings.warn(
                f"did not converge in {config.max_steps} steps; returning the tail-averaged "
                "iterate (raise max_steps or relax the stopping band).",
                ConvergenceWarning, stacklevel=2,
            )
        return self

    # ------------------------------------------------------------- learned views
    @property
    def estimates_(self):
        """``DataFrame`` indexed by param with columns ``coef`` / ``final`` / ``path_sd``.

        ``path_sd`` is the std of the parameter *path* over the tail window — an
        **optimisation-convergence diagnostic, NOT a standard error** (the estimator
        has no sampling-uncertainty story yet; there are deliberately no ``se``/``t``/``p``
        columns and this object is not an inferential ``summary``).
        """
        self._check_fitted()
        import pandas as pd

        paths = self.history_.param_history()
        window = max(self._config_used.convergence_window, self._config_used.stability_window)
        rows: dict[str, dict[str, float]] = {}
        for name in self.feature_names_:
            tail = paths.get(name, [])[-window:]
            path_sd = float(np.std(tail, ddof=0)) if len(tail) > 1 else 0.0
            rows[name] = {
                "coef": float(self.params_[name]),
                "final": float(self.params_final_[name]),
                "path_sd": path_sd,
            }
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=["coef", "final", "path_sd"])
        frame.index.name = "param"
        return frame

    def simulate(self, data: NetworkData | None = None, *, seed: int | None = None) -> Tensor:
        """Simulate ``Y`` at the estimated parameters (for sim-vs-obs plots)."""
        self._check_fitted()
        substrate = data.topology if data is not None else self._train_substrate
        if seed is not None:
            torch.manual_seed(int(seed))
        with torch.no_grad():
            return self.model_(substrate.W, substrate.X).detach()

    def discriminator_scores(
        self, data: NetworkData | None = None, *, n_roots: int = 512
    ) -> tuple[Tensor, Tensor]:
        """Return ``(real, fake)`` discriminator scores ``sigmoid(logit)`` for the score plot."""
        self._check_fitted()
        substrate = data.topology if data is not None else self._train_substrate
        y_obs = data.y if data is not None else self._train_y
        norm = substrate.make_norm_stats(y_obs)
        roots, _ = substrate.sample_roots(n_roots)
        with torch.no_grad():
            y_sim = self.model_(substrate.W, substrate.X)
            real_batch, real_idx = substrate.build_batch(roots, y_obs, norm, step=0, role="real")
            fake_batch, fake_idx = substrate.build_batch(roots, y_sim, norm, step=0, role="fake")
            real = torch.sigmoid(self.discriminator_(real_batch.x, real_batch.edge_index, real_idx))
            fake = torch.sigmoid(self.discriminator_(fake_batch.x, fake_batch.edge_index, fake_idx))
        return real, fake

    def recovery_table(self, true_params: Mapping[str, float]):
        """Convenience wrapper for :func:`adversarial_networks.reporting.recovery_table`."""
        self._check_fitted()
        from .reporting import recovery_table

        return recovery_table(self, true_params)

    # ------------------------------------------------------------ sklearn introspection
    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {
            "model": self.model, "discriminator": self.discriminator, "config": self.config,
            "max_steps": self.max_steps, "batch_size": self.batch_size, "lr_d": self.lr_d,
            "lr_g": self.lr_g, "seed": self.seed, "instance_noise": self.instance_noise,
            "gradient_transform": self.gradient_transform, "observers": self.observers,
            "device": self.device,
        }

    def set_params(self, **params: object) -> AdversarialEstimator:
        for key, value in params.items():
            setattr(self, key, value)
        return self
