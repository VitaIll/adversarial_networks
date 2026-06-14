"""Per-estimation configuration for the adversarial estimation engine.

``EstimatorConfig`` isolates the controls of a *single* estimation run — the
optimiser, batch size, discriminator/structural update ratio, learning-rate
schedule, the loss-band convergence criterion, and parameter-stability stopping —
from the *data-generating experiment* (``ExperimentConfig``) and the *Monte Carlo
orchestration* (``MonteCarloConfig``). This separation of concerns is the reason
the engine has one clean ``fit()`` surface: the same estimator runs under any
experiment and any runner.

For backwards compatibility, :meth:`EstimatorConfig.from_configs` derives an
``EstimatorConfig`` from the existing :class:`~src.config.ExperimentConfig` and
:class:`~src.config.MonteCarloConfig`, so the refactored Monte Carlo pipeline and
notebooks reuse their current configuration objects unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import ExperimentConfig, MonteCarloConfig


@dataclass(frozen=True)
class EstimatorConfig:
    """Controls for one adversarial estimation run.

    Attributes:
        max_steps: Hard cap on outer (structural) steps.
        min_steps: Earliest step at which convergence may be declared.
        batch_size: Requested root batch size per sampling call.
        n_disc: Discriminator updates per structural update (TTUR ratio).
        lr_d: Discriminator Adam learning rate.
        lr_g: Structural Adam learning rate.
        grad_clip_norm: Max L2 norm for the structural gradient (clipped before the
            optimiser step). The standard finite-sample stabiliser of Section 4.2.
        lr_g_decay_steps: Ascending step milestones at which ``lr_g`` is multiplied
            by ``lr_g_decay_factor``.
        lr_g_decay_factor: Multiplicative decay factor in ``(0, 1]``.
        convergence_window: Rolling window for the loss-band convergence check.
        convergence_delta_d: Tolerance band around ``2 log 2`` for the rolling
            discriminator loss.
        convergence_delta_g: Tolerance band around ``log 2`` for the rolling
            structural loss.
        convergence_std_d_max: Max rolling discriminator-loss std in the window.
        convergence_std_g_max: Max rolling structural-loss std in the window.
        stability_window: Trailing window over which parameter paths must be flat.
        stability_range_tol: Default max (max - min) range over the stability
            window for a parameter to count as stabilised.
        stability_range_tol_overrides: Per-parameter range-tolerance overrides as
            ``((name, tol), ...)`` (e.g. a looser band for ``sigma_sq``).
        nonsaturating: Use the non-saturating structural loss (eq. 4.2) if true,
            else the saturating minimax form.
        differentiation: Structural-gradient strategy applied to the model during
            ``fit`` — ``"unroll"`` (autograd through the executed Picard) or
            ``"implicit"`` (the implicit-function-theorem adjoint). ``None`` keeps
            whatever the model was constructed with (default ``"unroll"``).
        seed: Optional seed; when set, the estimator reseeds the torch RNG and the
            substrate's sampler RNG at the start of ``fit`` for reproducibility.
    """

    max_steps: int = 2000
    min_steps: int = 0
    batch_size: int = 64
    n_disc: int = 1
    lr_d: float = 2e-4
    lr_g: float = 3e-3
    grad_clip_norm: float = 10.0
    lr_g_decay_steps: tuple[int, ...] = ()
    lr_g_decay_factor: float = 1.0
    convergence_window: int = 100
    convergence_delta_d: float = 0.01
    convergence_delta_g: float = 0.01
    convergence_std_d_max: float = 0.1
    convergence_std_g_max: float = 0.1
    stability_window: int = 30
    stability_range_tol: float = 0.01
    stability_range_tol_overrides: tuple[tuple[str, float], ...] = ()
    nonsaturating: bool = True
    differentiation: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.min_steps < 0:
            raise ValueError(f"min_steps must be non-negative, got {self.min_steps}")
        if self.min_steps > self.max_steps:
            raise ValueError(
                f"min_steps must be <= max_steps, got {self.min_steps} > {self.max_steps}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_disc <= 0:
            raise ValueError(f"n_disc must be positive, got {self.n_disc}")
        if self.lr_d <= 0.0:
            raise ValueError(f"lr_d must be positive, got {self.lr_d}")
        if self.lr_g <= 0.0:
            raise ValueError(f"lr_g must be positive, got {self.lr_g}")
        if self.grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be positive, got {self.grad_clip_norm}")
        if any(step <= 0 for step in self.lr_g_decay_steps):
            raise ValueError("lr_g_decay_steps entries must be positive.")
        if tuple(sorted(self.lr_g_decay_steps)) != tuple(self.lr_g_decay_steps):
            raise ValueError("lr_g_decay_steps must be sorted ascending.")
        if not (0.0 < self.lr_g_decay_factor <= 1.0):
            raise ValueError(
                f"lr_g_decay_factor must satisfy 0 < factor <= 1, got {self.lr_g_decay_factor}"
            )
        if self.convergence_window <= 0:
            raise ValueError(f"convergence_window must be positive, got {self.convergence_window}")
        for name in ("convergence_delta_d", "convergence_delta_g", "convergence_std_d_max", "convergence_std_g_max"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.stability_window <= 0:
            raise ValueError(f"stability_window must be positive, got {self.stability_window}")
        if self.stability_range_tol <= 0.0:
            raise ValueError(f"stability_range_tol must be positive, got {self.stability_range_tol}")
        for name, tol in self.stability_range_tol_overrides:
            if tol <= 0.0:
                raise ValueError(f"stability override for {name!r} must be positive, got {tol}")
        if self.differentiation not in (None, "unroll", "implicit"):
            raise ValueError(
                "differentiation must be None, 'unroll', or 'implicit', got "
                f"{self.differentiation!r}"
            )

    def override_tol_for(self, param_name: str) -> float:
        """Return the stability range tolerance for a named parameter."""
        for name, tol in self.stability_range_tol_overrides:
            if name == param_name:
                return float(tol)
        return float(self.stability_range_tol)

    @classmethod
    def recovery_default(cls) -> EstimatorConfig:
        """The calibrated fast-scale config that recovers parameters at ~10k BA.

        Calibrated against the linear-in-means recovery study (10k Barabasi-Albert,
        ``beta_cap=0.85``, discriminator ``hidden_dim=12``/``logit_clip=4.0``/``num_layers=2``)
        over multiple seeds. **The recovery recipe also needs the instance-noise blur**
        — pair this config with
        ``InstanceNoiseConfig(enabled=True, tau_x0=1.0, tau_y0=1.0, schedule="linear", anneal_steps=2000)``:
        the slow (2000-step) blur anneal keeps the discriminator from sharpening while
        the ``max_steps=1200`` horizon stops before the late-training overshoot, so the
        tail-averaged estimate sits near the truth. Recovery is rough at this fast scale
        (``beta`` — the social multiplier — is biased low; ``sigma^2`` is biased at finite
        ``n`` and is not asserted); the asymptotic (paper-scale) regime recovers tightly.
        The recovery test asserts ``beta``/``gamma`` within the observed spread.
        """
        return cls(
            max_steps=1200,
            min_steps=700,
            batch_size=17,
            n_disc=1,
            lr_d=2e-4,
            lr_g=3e-3,
            grad_clip_norm=10.0,
            lr_g_decay_steps=(220, 420, 620, 780),
            lr_g_decay_factor=1.0,
            convergence_window=100,
            convergence_delta_d=0.01,
            convergence_delta_g=0.01,
            convergence_std_d_max=0.1,
            convergence_std_g_max=0.1,
            stability_window=30,
            stability_range_tol=0.01,
            stability_range_tol_overrides=(("gamma", 0.01), ("sigma_sq", 0.1), ("mu", 0.05)),
            nonsaturating=True,
            differentiation=None,
            seed=0,
        )

    @classmethod
    def from_configs(
        cls,
        experiment: ExperimentConfig,
        monte_carlo: MonteCarloConfig,
        *,
        seed: int | None = None,
    ) -> EstimatorConfig:
        """Adapt the legacy ``ExperimentConfig`` + ``MonteCarloConfig`` to an ``EstimatorConfig``.

        Mirrors the optimisation choices currently hard-coded in
        ``experiments/asymptotic_mc_experiment.py`` (notably that the generator
        gradient clip comes from ``MonteCarloConfig.grad_clip_norm``), so the
        refactored pipeline reproduces the existing training dynamics exactly.
        """
        training = experiment.training
        max_steps = int(monte_carlo.max_steps) if monte_carlo.max_steps is not None else int(training.n_steps)
        min_steps = int(monte_carlo.min_steps) if monte_carlo.min_steps is not None else 0
        overrides = (
            ("gamma", float(monte_carlo.stability_gamma_range_tol)),
            ("sigma_sq", float(monte_carlo.stability_sigma_sq_range_tol)),
        )
        return cls(
            max_steps=max_steps,
            min_steps=min(min_steps, max_steps),
            batch_size=int(training.batch_size),
            n_disc=int(training.n_disc),
            lr_d=float(training.lr_d),
            lr_g=float(training.lr_g),
            grad_clip_norm=float(monte_carlo.grad_clip_norm),
            lr_g_decay_steps=tuple(int(s) for s in monte_carlo.lr_g_decay_steps),
            lr_g_decay_factor=float(monte_carlo.lr_g_decay_factor),
            convergence_window=int(monte_carlo.convergence_window),
            convergence_delta_d=float(monte_carlo.convergence_delta_d),
            convergence_delta_g=float(monte_carlo.convergence_delta_g),
            convergence_std_d_max=float(monte_carlo.convergence_std_d_max),
            convergence_std_g_max=float(monte_carlo.convergence_std_g_max),
            stability_window=int(monte_carlo.stability_window),
            stability_range_tol=float(monte_carlo.stability_beta_range_tol),
            stability_range_tol_overrides=overrides,
            seed=seed,
        )
