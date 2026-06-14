"""Typed contracts for the adversarial estimation engine.

This module defines the structural interfaces (``Protocol`` types) and immutable
value objects on which the estimation engine depends. Collecting them in one
place keeps caller/callable surfaces exact and predictable:

* A :class:`StructuralModel` promises exactly ``simulate + get_params + nn.Module``.
* A :class:`TestFunction` (discriminator / adaptive test function) promises
  ``logits = D(x, edge_index, root_indices)``.
* The engine promises to emit a :class:`StepMetrics` per outer step and to return
  an :class:`EstimationResult`.
* A :class:`MetricsObserver` consumes those records; nothing downstream relies on
  behaviour that is not stated here.

The protocols are ``runtime_checkable`` so the engine can perform a *soft*
conformance check at its boundary (presence of the required methods) and fail
loudly with an attributable message, rather than failing deep inside the loop.

References:
    Illichmann & Zacchia (2026), *Adversarial Structural Estimation on Graphs*,
    Algorithm 1 (alternating minimax) and Section 4.2 (training mechanics).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from torch import Tensor
from torch.nn import Parameter


@runtime_checkable
class StructuralModel(Protocol):
    """A differentiable structural simulator over a single fixed graph.

    A ``StructuralModel`` maps the (row-stochastic) interaction matrix ``W`` and
    the covariate vector ``X`` to a simulated equilibrium outcome ``Y`` with
    fresh structural shocks, differentiably in its leaf parameters. It is an
    ``nn.Module`` (so it owns its parameters, dtype and device) and reports its
    current *constrained* structural parameters through :meth:`get_params`.

    Contract:
        * ``model(W, X) -> Y`` where ``W`` is a sparse ``(n, n)`` COO tensor,
          ``X`` is a dense ``(n,)`` float tensor on the same device, and ``Y`` is
          a dense ``(n,)`` float tensor carrying gradients to the leaf params.
        * ``get_params()`` returns a mapping from human-readable parameter name
          to its current constrained scalar value (no gradient). The set of keys
          is fixed for a given model instance but is *not* assumed by the engine,
          so the engine is model-agnostic over ``{beta, gamma, sigma_sq}``,
          ``{gamma, lambda_, mu, r, sigma_sq}``, or any future game.

    Both :class:`adversarial_networks.generators.LinearInMeansGenerator` and
    :class:`adversarial_networks.generators.EffortGameGenerator` satisfy this
    protocol structurally (via the ``NetworkGameGenerator`` base). New network
    games plug in by implementing the same surface.
    """

    def __call__(self, W: Tensor, X: Tensor) -> Tensor: ...

    def get_params(self) -> dict[str, float]: ...

    def parameters(self) -> Iterator[Parameter]: ...

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]: ...


@runtime_checkable
class TestFunction(Protocol):
    """An adaptive test function (discriminator) over rooted ego objects.

    Maps batched node features ``x`` of shape ``(num_nodes, d)``, an ``edge_index``
    of shape ``(2, num_edges)``, and ``root_indices`` of shape ``(batch,)`` to a
    vector of per-root logits (pre-sigmoid). The paper's adaptive test function
    ``D_phi`` and the GAN discriminator are the same object; positive logits push
    the implied score ``sigmoid(logit)`` towards classifying the input as
    observed ("real").
    """

    def __call__(self, x: Tensor, edge_index: Tensor, root_indices: Tensor) -> Tensor: ...

    def parameters(self) -> Iterator[Parameter]: ...

    def named_parameters(self) -> Iterator[tuple[str, Parameter]]: ...


@dataclass(frozen=True)
class StepMetrics:
    """Immutable record of one outer (structural-parameter) optimisation step.

    This is the single observability unit emitted by the engine. It is
    deliberately model-agnostic: ``params``/``param_grads`` carry whatever the
    model's :meth:`StructuralModel.get_params` exposes, and ``extras`` is an open
    slot for diagnostics added by later milestones (e.g. the Fisher
    condition number and per-parameter gradient SNR for objective 7).

    Attributes:
        step: 1-based outer step index.
        params: Current constrained structural parameters.
        loss_d: Discriminator minibatch loss at this step (``-> 2 log 2`` at the
            population optimum).
        loss_g: Generator (structural) minibatch loss (``-> log 2`` at the optimum).
        loss_d_rolling: Rolling-mean discriminator loss over the convergence
            window (``nan`` until the window fills).
        loss_g_rolling: Rolling-mean generator loss (``nan`` until filled).
        grad_norm_g: Total L2 norm of the structural gradient *before* clipping.
        picard_iterations: Picard iterations used in the structural-phase solve.
        roots_requested: Root batch size requested from the sampler.
        roots_achieved: Root batch size actually returned (``< requested`` only
            under disjoint packing shortfalls).
        tau_x: Instance-noise blur std applied to covariates this step.
        tau_y: Instance-noise blur std applied to outcomes this step.
        in_equilibrium: Whether the loss-band convergence check passed this step.
        lr_g: Effective structural learning rate this step (after any decay).
        newton_iterations: Max Newton iterations used (nonlinear games only).
        sampler_radius: Disjoint-exclusion radius actually used (``None`` for
            uniform sampling).
        param_grads: Per-leaf-parameter gradient component (raw-parameter space),
            for gradient-SNR diagnostics; ``None`` when not collected.
        extras: Open diagnostic slot (e.g. ``fisher_condition_number``).
    """

    step: int
    params: Mapping[str, float]
    loss_d: float
    loss_g: float
    loss_d_rolling: float
    loss_g_rolling: float
    grad_norm_g: float
    picard_iterations: int
    roots_requested: int
    roots_achieved: int
    tau_x: float
    tau_y: float
    in_equilibrium: bool
    lr_g: float
    newton_iterations: int | None = None
    sampler_radius: int | None = None
    param_grads: Mapping[str, float] | None = None
    extras: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EstimationResult:
    """Immutable outcome of a single estimation run (one realisation).

    Attributes:
        status: ``"ok"`` on a completed run, or ``"failed:<reason>"``.
        converged: Whether the loss-band + parameter-stability stopping rule
            fired before the step cap.
        final_step: Outer step index at termination.
        params: Tail-averaged point estimate (mean of the constrained parameter
            paths over the trailing stability/convergence window).
        params_final: Last-iterate constrained parameters.
        loss_d_rolling_final: Trailing rolling-mean discriminator loss.
        loss_g_rolling_final: Trailing rolling-mean generator loss.
        n_steps_run: Number of outer steps actually executed.
        failure_reason: Populated only when ``status`` begins with ``"failed:"``.
        extras: Open slot for run-level diagnostics.
    """

    status: str
    converged: bool
    final_step: int
    params: Mapping[str, float]
    params_final: Mapping[str, float]
    loss_d_rolling_final: float
    loss_g_rolling_final: float
    n_steps_run: int
    failure_reason: str = ""
    extras: Mapping[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the run completed without a structural failure."""
        return self.status == "ok"


class MetricsObserver(Protocol):
    """Sink for engine observability events.

    Observers are notified at run start, after every outer step, and at run end.
    Implementations must be cheap and side-effect isolated: an observer raising
    must not corrupt the estimation, so the engine guards observer dispatch.
    """

    def on_run_start(self, meta: Mapping[str, object]) -> None: ...

    def on_step(self, metrics: StepMetrics) -> None: ...

    def on_run_end(self, result: EstimationResult) -> None: ...
