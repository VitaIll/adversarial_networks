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

    expected_in_dim = substrate.d_x + 2
    disc_in_dim = getattr(discriminator, "in_dim", None)
    if disc_in_dim is not None and int(disc_in_dim) != expected_in_dim:
        raise ValueError(
            f"discriminator in_dim ({disc_in_dim}) must equal d_x + 2 ({expected_in_dim}) "
            f"for the substrate's covariate width d_x={substrate.d_x}; the node feature "
            "row is [X_tilde (d_x), Y_tilde, root_marker]."
        )

    # C6(i) boundary observability: the test function must be clipped to a fixed
    # [eta, 1-eta] band. A discriminator that exposes logit_clip is None has clipping
    # disabled and so optimises the eta=0 criterion, for which the paper gives no
    # consistency guarantee. We warn only when the attribute is present AND None: a
    # custom TestFunction that does not expose it may clip internally, so its absence
    # is not actionable. (RootedMPNNDiscriminator now DEFAULTS to a positive clip (5.0);
    # logit_clip=None is an explicit opt-out of the C6(i) bound — by the default
    # discriminator or a custom unclipped TestFunction — which this surfaces once at the
    # start of a run.)
    if getattr(discriminator, "logit_clip", "unknown") is None:
        warnings.warn(
            "discriminator has logit_clip=None (clipping disabled): C6 requires "
            "D in [eta, 1-eta] for a fixed eta in (0, 1/2) so the per-object loss "
            "|log D| is bounded, and an unclipped discriminator instead optimises the "
            "eta=0 criterion (no consistency guarantee). Set a positive logit_clip "
            "(e.g. the default 5.0) to restore the C6 bound.",
            RuntimeWarning,
            stacklevel=2,
        )

    norm_stats = substrate.make_norm_stats(Y_obs)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d)
    opt_g = torch.optim.Adam(model.parameters(), lr=config.lr_g)
    stopping = StoppingRule(config)
    decay_steps = {int(s) for s in config.lr_g_decay_steps}
    observers = list(observers)

    # Consistency guard: the instance-noise blur must anneal to zero over the WHOLE
    # tail-averaging window so the estimator targets the original (unblurred) criterion.
    # The point estimate is the tail average over the trailing window (see _finalize), so
    # the relevant step is the FIRST step of that window, not the terminal step: a blur
    # still positive at the window start contaminates the average even if it has reached
    # zero by max_steps. A residual blur (min_tau>0, anneal_steps reaching into the tail,
    # or a constant/exp schedule) means the criterion never converges to the original one,
    # biasing the tail-averaged estimate.
    if instance_noise is not None and bool(instance_noise.enabled):
        tail_window = max(config.convergence_window, config.stability_window)
        tail_start_step = max(1, config.max_steps - tail_window + 1)
        residual_tau = compute_instance_noise_taus(
            instance_noise, generator_step=tail_start_step
        )
        if residual_tau > 0.0:
            exp_note = (
                " Note: while annealing (step < anneal_steps) the 'exp' schedule is "
                "asymptotic and never reaches exactly zero, so an anneal reaching into the "
                "tail window leaves residual blur there; either set anneal_steps <= "
                "max_steps-tail_window (the exp branch then snaps to exactly min_tau before "
                "the tail window) or use the linear schedule, which the consistency "
                "guarantee assumes."
                if str(instance_noise.schedule) == "exp"
                else ""
            )
            warnings.warn(
                "instance_noise blur does not reach zero by the start of the tail-averaging "
                f"window (residual tau_y={residual_tau:.4g} at step "
                f"{tail_start_step}=max_steps-tail_window+1, tail_window={tail_window}, "
                f"max_steps={config.max_steps}); the estimator targets a residually-blurred "
                "(non-consistent) criterion and the tail-averaged estimate is biased. Use the "
                "linear schedule with min_tau=0.0 and anneal_steps<=max_steps-tail_window so "
                "the blur reaches exactly zero before the tail-averaging window begins."
                + exp_note,
                RuntimeWarning,
                stacklevel=2,
            )

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

    # Solver non-convergence observability: a forward whose Picard (or Newton) solve
    # exhausts its cap WITHOUT the tolerance test firing returns a truncated,
    # possibly-non-equilibrium iterate, which would then feed the discriminator silently.
    # We key cap-hit detection off the solver's own ``converged`` flag (NOT iters>=cap,
    # which false-positives on a genuine last-iteration stop and is residual-blind), warn
    # the first time each solver fails to converge, surface the residual, and report the
    # totals in EstimationResult.extras. ``newton_cap_hits`` is only meaningful for models
    # that actually run Newton (a closed-form best_response never does), so it is tracked
    # only once Newton has actually executed (last_newton_max_iters > 0 at some step).
    picard_max = getattr(model, "picard_max", None)
    picard_cap_hits = 0
    newton_cap_hits = 0
    ran_newton = False
    # Sampler-shortfall observability: a disjoint packer that returns fewer roots
    # than requested (or falls back to uniform) weakens the near-independence the
    # disjoint modes rely on. The structural- and discriminator-phase draws are tracked
    # separately (each is an independent stochastic packing), and a realised exclusion
    # radius below 2*k forfeits vertex-disjointness even when the batch fills. Warn the
    # first time each condition occurs, matching the cap-hit style.
    sampler_shortfall_warned = False
    disc_sampler_shortfall_warned = False
    sub_two_k_radius_warned = False

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
            roots, disc_root_result = substrate.sample_roots(config.batch_size)
            # The disc-phase draws drive the discriminator gradient update directly and
            # each greedy packing is stochastic per random permutation, so one can
            # under-fill independently of the structural draw; surface it once.
            if not disc_sampler_shortfall_warned and (
                not disc_root_result.met_target or disc_root_result.fallback_reason
            ):
                reason = disc_root_result.fallback_reason or "below_requested_batch"
                warnings.warn(
                    f"Discriminator-phase root sampler shortfall at step {step}: requested "
                    f"{int(disc_root_result.requested_size)} roots but achieved "
                    f"{int(disc_root_result.achieved_size)} "
                    f"(mode={disc_root_result.mode!r}, reason={reason!r}); the packed egos "
                    "driving the discriminator update are less independent than the disjoint "
                    "mode assumes. Lower batch_size, relax the exclusion radius, or switch "
                    "root_sampler_mode.",
                    RuntimeWarning, stacklevel=2,
                )
                disc_sampler_shortfall_warned = True
            sub_two_k_radius_warned = _maybe_warn_sub_two_k(
                disc_root_result, k=substrate.k, step=step, phase="discriminator",
                already_warned=sub_two_k_radius_warned,
            )
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
        # A disjoint-packing shortfall (fewer roots than requested, or a fallback to
        # uniform) is otherwise invisible; surface it once with the requested vs
        # achieved size and the fallback reason.
        if not sampler_shortfall_warned and (
            not root_result.met_target or root_result.fallback_reason
        ):
            reason = root_result.fallback_reason or "below_requested_batch"
            warnings.warn(
                f"Root sampler shortfall at step {step}: requested "
                f"{int(root_result.requested_size)} roots but achieved "
                f"{int(root_result.achieved_size)} "
                f"(mode={root_result.mode!r}, reason={reason!r}); the packed egos are "
                "less independent than the disjoint mode assumes. Lower batch_size, "
                "relax the exclusion radius, or switch root_sampler_mode.",
                RuntimeWarning, stacklevel=2,
            )
            sampler_shortfall_warned = True
        # A realised exclusion radius below 2*k forfeits vertex-disjointness (the egos can
        # share vertices) even when the batch fills, so the disjoint mode's near-
        # independence is silently lost; surface it once, keyed on the independence
        # property rather than batch-fill (the existing shortfall gate cannot catch it).
        sub_two_k_radius_warned = _maybe_warn_sub_two_k(
            root_result, k=substrate.k, step=step, phase="structural",
            already_warned=sub_two_k_radius_warned,
        )
        Y_sim = simulate(detached=False)
        if not torch.isfinite(Y_sim).all():
            return failure(step, "Y_sim_non_finite_G")

        # Non-convergence detection on the on-tape (gradient-carrying) forward: the iterate
        # that actually feeds the discriminator this step. Keyed off the solver's own
        # ``converged`` flag (the tol test fired) — NOT iters>=cap, which false-positives on
        # a genuine last-iteration stop and is residual-blind. Warn once, surface the
        # residual, then keep counting.
        last_picard_residual = float(getattr(model, "last_picard_residual", 0.0))
        picard_converged = bool(getattr(model, "last_picard_converged", True))
        if picard_max is not None and not picard_converged:
            if picard_cap_hits == 0:
                warnings.warn(
                    f"Picard did not converge (hit cap picard_max={picard_max}, residual "
                    f"max|Y_t+1-Y_t|={last_picard_residual:.4g} >= picard_tol) at step {step}: "
                    "a non-equilibrium outcome is feeding the discriminator. Raise picard_max "
                    "(high-contraction rho near 1 needs a larger cap) or verify contraction "
                    "(rho < 1) via check_model.",
                    RuntimeWarning, stacklevel=2,
                )
            picard_cap_hits += 1
        # Newton diagnostics are meaningful only for a model that actually runs Newton: a
        # closed-form best_response game never does. last_newton_max_iters stays 0 unless
        # Newton executed, so this gate (D6-R2) is exact — 0 for LinearInMeans, >0 for the
        # effort game / any best_response that calls newton_solve.
        last_newton_iters = int(getattr(model, "last_newton_max_iters", 0))
        if last_newton_iters > 0:
            ran_newton = True
            newton_max = int(getattr(model, "newton_max", 0))
            last_newton_residual = float(getattr(model, "last_newton_residual", 0.0))
            newton_converged = bool(getattr(model, "last_newton_converged", True))
            if newton_max > 0 and not newton_converged:
                if newton_cap_hits == 0:
                    warnings.warn(
                        f"Newton did not converge (hit cap newton_max={newton_max}, residual "
                        f"max|delta|={last_newton_residual:.4g} >= newton_tol) at step {step}: "
                        "the per-node best response is unconverged and a non-equilibrium "
                        "outcome is feeding the discriminator. Raise newton_max or verify the "
                        "FOC is well-conditioned via check_model.",
                        RuntimeWarning, stacklevel=2,
                    )
                newton_cap_hits += 1

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
        tau_y = compute_instance_noise_taus(instance_noise, generator_step=step)

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
            tau_y=float(tau_y), in_equilibrium=decision.in_equilibrium,
            lr_g=lr_g, newton_iterations=_maybe_int(getattr(model, "last_newton_max_iters", None)),
            sampler_radius=root_result.radius_used,
            picard_residual=last_picard_residual,
            picard_converged=picard_converged,
            sampler_met_target=bool(root_result.met_target),
            sampler_fallback_reason=str(root_result.fallback_reason),
            extras=extras,
        ))

        if decision.converged:
            converged = True
            break

    result = _finalize(
        converged=converged, final_step=final_step, config=config,
        param_history=param_history, loss_d_history=loss_d_history, loss_g_history=loss_g_history,
        picard_cap_hits=picard_cap_hits,
        # newton_cap_hits is surfaced only for models that actually ran Newton (D6-R2): a
        # closed-form best_response model must not report a meaningless Newton diagnostic.
        newton_cap_hits=(newton_cap_hits if ran_newton else None),
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
    picard_cap_hits: int = 0,
    newton_cap_hits: int | None = None,
    extras: Mapping[str, float] | None = None,
) -> EstimationResult:
    """Assemble the tail-averaged estimate and final diagnostics.

    ``picard_cap_hits`` / ``newton_cap_hits`` are surfaced in
    :attr:`EstimationResult.extras` (a non-equilibrium outcome silently feeding the
    discriminator is otherwise invisible); the Newton key is added only when the model
    actually ran Newton during the run (``newton_cap_hits is not None``). Any
    caller-supplied ``extras`` are merged, not clobbered.
    """
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

    run_extras: dict[str, float] = dict(extras or {})
    run_extras["picard_cap_hits"] = int(picard_cap_hits)
    if newton_cap_hits is not None:
        run_extras["newton_cap_hits"] = int(newton_cap_hits)

    return EstimationResult(
        status="ok", converged=converged, final_step=int(final_step),
        params=point_estimate, params_final=final_params,
        loss_d_rolling_final=loss_d_rolling_final, loss_g_rolling_final=loss_g_rolling_final,
        n_steps_run=n_logged, extras=run_extras,
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


def _maybe_warn_sub_two_k(
    root_result, *, k: int, step: int, phase: str, already_warned: bool
) -> bool:
    """Warn once when a realised disjoint exclusion radius is below ``2*k``.

    Two radius-``k`` ego balls are vertex-disjoint iff their centres are more than ``2k``
    apart (fn. 26), so a realised ``radius_used < 2*k`` means the sampled egos can share
    vertices — the disjoint mode's near-independence is forfeited even when the batch
    fills (the batch-fill shortfall gate cannot catch this). The estimator owns ``k`` and
    so makes the check the sampler cannot. Returns the updated ``already_warned`` flag.
    """
    if already_warned:
        return True
    radius_used = getattr(root_result, "radius_used", None)
    if radius_used is not None and int(radius_used) < 2 * int(k):
        warnings.warn(
            f"{phase.capitalize()}-phase root sampler used exclusion radius "
            f"{int(radius_used)} < 2*k = {2 * int(k)} (k={int(k)}) at step {step}: the "
            "sampled radius-k egos are NOT vertex-disjoint (they can share vertices), so "
            "the near-independence the disjoint packing relies on does not hold even "
            "though the batch filled. Raise the exclusion radius (or relax-ladder rungs) "
            f"to >= {2 * int(k)} for vertex-disjoint egos.",
            RuntimeWarning, stacklevel=2,
        )
        return True
    return False


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

        expected_in_dim = data.topology.d_x + 2
        disc_in_dim = getattr(self.discriminator, "in_dim", None)
        if disc_in_dim is not None and int(disc_in_dim) != expected_in_dim:
            raise ValueError(
                f"discriminator in_dim ({disc_in_dim}) must equal d_x + 2 ({expected_in_dim}) "
                f"for the data's covariate width d_x={data.topology.d_x}; the test function "
                "input width must match the [X_tilde (d_x), Y_tilde, root_marker] node row."
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
