"""Configuration dataclasses for experiment parameters.

All configuration is type-safe, validated, and serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class GraphConfig:
    """Graph-generation controls that determine topology and overlap propensity."""

    n_nodes: int = 250000
    """Target number of nodes in the generated graph."""

    graph_type: Literal["lfr", "ba"] = "lfr"
    """Graph family: `lfr` (community structure) or `ba` (preferential attachment)."""

    ba_m: int = 3
    """BA only: edges added by each new node; larger values create denser local neighborhoods."""

    lfr_tau1: float = 2.5
    """LFR only: power-law exponent for node degrees (lower means heavier-tailed degrees)."""

    lfr_tau2: float = 1.5
    """LFR only: power-law exponent for community sizes (lower means more size inequality)."""

    lfr_mu: float = 0.3
    """LFR only: fraction of each node's edges that cross communities (higher means weaker separation)."""

    lfr_average_degree: int | None = 6
    """LFR only: target average degree; set exactly one of this or `lfr_min_degree`."""

    lfr_min_degree: int | None = None
    """LFR only: minimum degree; set exactly one of this or `lfr_average_degree`."""

    lfr_min_community: int = 20
    """LFR only: lower bound on community sizes used by the benchmark generator."""

    lfr_max_community: int = 100
    """LFR only: upper bound on community sizes used by the benchmark generator."""

    lfr_max_degree: int | None = 100
    """LFR only: optional upper bound on degree to limit extreme hubs."""

    lfr_max_iters: int = 500
    """LFR only: max internal retries per generation attempt before the call fails."""

    seed: int = 42
    """Seed controlling graph construction and reproducible topology draws."""

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_nodes <= 0:
            raise ValueError(f"n_nodes must be positive, got {self.n_nodes}")
        if self.graph_type not in ("lfr", "ba"):
            raise ValueError(
                f"graph_type must be 'lfr' or 'ba', got {self.graph_type!r}"
            )
        if self.ba_m <= 0 or self.ba_m >= self.n_nodes:
            raise ValueError(f"ba_m must satisfy 0 < ba_m < n_nodes, got {self.ba_m}")
        if self.lfr_tau1 <= 1.0:
            raise ValueError(f"lfr_tau1 must be > 1, got {self.lfr_tau1}")
        if self.lfr_tau2 <= 1.0:
            raise ValueError(f"lfr_tau2 must be > 1, got {self.lfr_tau2}")
        if not (0.0 <= self.lfr_mu <= 1.0):
            raise ValueError(f"lfr_mu must satisfy 0 <= mu <= 1, got {self.lfr_mu}")
        if (self.lfr_average_degree is None) == (self.lfr_min_degree is None):
            raise ValueError(
                "Exactly one of lfr_average_degree or lfr_min_degree must be provided."
            )
        if self.lfr_average_degree is not None and self.lfr_average_degree <= 0:
            raise ValueError(
                "lfr_average_degree must be positive when provided, "
                f"got {self.lfr_average_degree}"
            )
        if self.lfr_min_degree is not None and self.lfr_min_degree <= 0:
            raise ValueError(
                "lfr_min_degree must be positive when provided, "
                f"got {self.lfr_min_degree}"
            )
        if self.lfr_min_community <= 0:
            raise ValueError(
                f"lfr_min_community must be positive, got {self.lfr_min_community}"
            )
        if self.lfr_max_community <= 0:
            raise ValueError(
                f"lfr_max_community must be positive, got {self.lfr_max_community}"
            )
        if self.lfr_max_community < self.lfr_min_community:
            raise ValueError(
                "lfr_max_community must be >= lfr_min_community, got "
                f"{self.lfr_max_community} < {self.lfr_min_community}"
            )
        if self.lfr_max_community > self.n_nodes:
            raise ValueError(
                f"lfr_max_community must be <= n_nodes, got {self.lfr_max_community}"
            )
        if self.lfr_max_degree is not None and self.lfr_max_degree <= 0:
            raise ValueError(
                "lfr_max_degree must be positive when provided, "
                f"got {self.lfr_max_degree}"
            )
        if (
            self.lfr_max_degree is not None
            and self.lfr_average_degree is not None
            and self.lfr_max_degree < self.lfr_average_degree
        ):
            raise ValueError(
                "lfr_max_degree must be >= lfr_average_degree, got "
                f"{self.lfr_max_degree} < {self.lfr_average_degree}"
            )
        if (
            self.lfr_max_degree is not None
            and self.lfr_min_degree is not None
            and self.lfr_max_degree < self.lfr_min_degree
        ):
            raise ValueError(
                "lfr_max_degree must be >= lfr_min_degree, got "
                f"{self.lfr_max_degree} < {self.lfr_min_degree}"
            )
        if self.lfr_max_iters <= 0:
            raise ValueError(f"lfr_max_iters must be positive, got {self.lfr_max_iters}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass(frozen=True)
class ModelConfig:
    """Structural-model and discriminator architecture controls."""

    k: int = 2
    """Ego radius used for extracted rooted subgraphs; larger `k` increases local context and overlap."""

    beta_cap: float = 1.0
    """Upper bound on |beta| via reparameterization; lower values enforce stronger contraction."""

    picard_tol: float = 1e-6
    """Picard stopping tolerance; smaller values run more iterations for tighter fixed-point solves."""

    picard_max: int = 100
    """Hard cap on Picard iterations per forward pass."""

    hidden_dim: int = 64
    """Width of discriminator GNN/MLP layers (capacity vs. compute)."""

    logit_clip: float = 10.0
    """Absolute clipping bound for discriminator logits to stabilize adversarial losses."""

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if not (0.0 < self.beta_cap <= 1.0):
            raise ValueError(f"beta_cap must satisfy 0 < beta_cap <= 1, got {self.beta_cap}")
        if self.picard_tol <= 0.0:
            raise ValueError(f"picard_tol must be positive, got {self.picard_tol}")
        if self.picard_max <= 0:
            raise ValueError(f"picard_max must be positive, got {self.picard_max}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.logit_clip <= 0.0:
            raise ValueError(f"logit_clip must be positive, got {self.logit_clip}")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and root-sampling controls used during GAN training."""

    n_steps: int = 800
    """Total outer training steps (one generator update per step)."""

    batch_size: int = 64
    """Requested number of root nodes per sampling call (uniform returns exactly this many)."""

    n_disc: int = 5
    """Number of discriminator updates per generator update (TTUR ratio)."""
    lr_d: float = 1e-3
    """Discriminator optimizer learning rate."""

    lr_g: float = 5e-3
    """Generator optimizer learning rate."""

    grad_clip_norm_g: float = 5.0
    """Max norm for generator gradient clipping before the optimizer step."""

    root_sampler_mode: (
        Literal["uniform", "disjoint_once", "disjoint_best_of_k", "disjoint_relax"] | None
    ) = "disjoint_best_of_k"
    """Preferred root sampler mode.

    When `None`, legacy knobs (`root_sampling_scheme`, `relax_to_r2`, etc.) are
    resolved for backwards compatibility.
    """

    root_exclusion_r: int = 4
    """Exclusion radius for disjoint sampling, enforcing pairwise dist(u, v) > r."""

    disjoint_restarts_k: int | None = 50
    """Number of independent greedy rescans in best-of-k disjoint modes.

    If `None`, defaults are resolved by mode:
    - explicit modern modes: `3`
    - legacy `greedy_packing`: `1` (preserves historical behavior)
    """

    disjoint_min_batch: int | None = 64
    """Minimum accepted packed roots for `disjoint_relax`.

    If `None`, falls back to `min_roots_per_call` for backwards compatibility.
    """

    disjoint_relax_sequence: tuple[int, ...] | None = (3,2)
    """Radius ladder for `disjoint_relax`, e.g. `(3, 2)`.

    If `None`, defaults are resolved from legacy controls:
    `(root_exclusion_r, 2)` when `relax_to_r2` is enabled and `r != 2`, else `(r,)`.
    """

    disjoint_fallback: Literal["uniform", "best", "raise"] = "best"
    """Fallback policy for `disjoint_relax` when no radius reaches `disjoint_min_batch`."""

    root_sampling_scheme: Literal["uniform", "greedy_packing"] = "greedy_packing"
    """Legacy root sampler selector retained for compatibility."""

    min_roots_per_call: int = 64
    """Legacy minimum packed roots before fallback logic."""

    relax_to_r2: bool = False
    """Legacy flag enabling a weaker `r=2` retry when greedy packing under-fills."""

    mix_p_uniform: float = 0.0
    """Probability of forcing a uniform draw even in disjoint modes."""

    def resolved_root_sampler_mode(
        self,
    ) -> Literal["uniform", "disjoint_once", "disjoint_best_of_k", "disjoint_relax"]:
        """Resolve modern root sampler mode with legacy compatibility mapping."""
        if self.root_sampler_mode is not None:
            return self.root_sampler_mode
        if self.root_sampling_scheme == "uniform":
            return "uniform"
        return "disjoint_relax"

    def resolved_disjoint_restarts_k(self) -> int:
        """Resolve disjoint restart count with backwards-compatible defaults."""
        if self.disjoint_restarts_k is not None:
            return int(self.disjoint_restarts_k)
        if self.root_sampler_mode is None and self.root_sampling_scheme == "greedy_packing":
            return 1
        return 3

    def resolved_disjoint_min_batch(self) -> int:
        """Resolve minimum accepted packed roots for relax mode."""
        if self.disjoint_min_batch is not None:
            return int(self.disjoint_min_batch)
        return int(self.min_roots_per_call)

    def resolved_disjoint_relax_sequence(self) -> tuple[int, ...]:
        """Resolve radius ladder for relax mode with compatibility defaults."""
        if self.disjoint_relax_sequence is not None:
            return tuple(int(radius) for radius in self.disjoint_relax_sequence)
        if self.relax_to_r2 and self.root_exclusion_r != 2:
            return (int(self.root_exclusion_r), 2)
        return (int(self.root_exclusion_r),)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_disc <= 0:
            raise ValueError(f"n_disc must be positive, got {self.n_disc}")
        if self.lr_d <= 0.0:
            raise ValueError(f"lr_d must be positive, got {self.lr_d}")
        if self.lr_g <= 0.0:
            raise ValueError(f"lr_g must be positive, got {self.lr_g}")
        if self.grad_clip_norm_g <= 0.0:
            raise ValueError(
                f"grad_clip_norm_g must be positive, got {self.grad_clip_norm_g}"
            )
        if self.root_sampler_mode is not None and self.root_sampler_mode not in {
            "uniform",
            "disjoint_once",
            "disjoint_best_of_k",
            "disjoint_relax",
        }:
            raise ValueError(
                "root_sampler_mode must be one of {'uniform', 'disjoint_once', "
                "'disjoint_best_of_k', 'disjoint_relax'} when provided, got "
                f"{self.root_sampler_mode!r}"
            )
        if self.root_sampling_scheme not in ("uniform", "greedy_packing"):
            raise ValueError(
                "root_sampling_scheme must be 'uniform' or 'greedy_packing', got "
                f"{self.root_sampling_scheme!r}"
            )
        if self.root_exclusion_r < 0:
            raise ValueError(
                f"root_exclusion_r must be non-negative, got {self.root_exclusion_r}"
            )
        if self.min_roots_per_call <= 0:
            raise ValueError(
                "min_roots_per_call must be positive, "
                f"got {self.min_roots_per_call}"
            )
        if self.min_roots_per_call > self.batch_size:
            raise ValueError(
                "min_roots_per_call must be <= batch_size, got "
                f"{self.min_roots_per_call} > {self.batch_size}"
            )
        if self.disjoint_restarts_k is not None and self.disjoint_restarts_k <= 0:
            raise ValueError(
                "disjoint_restarts_k must be positive when provided, got "
                f"{self.disjoint_restarts_k}"
            )
        if self.disjoint_min_batch is not None and self.disjoint_min_batch <= 0:
            raise ValueError(
                "disjoint_min_batch must be positive when provided, got "
                f"{self.disjoint_min_batch}"
            )
        if self.disjoint_min_batch is not None and self.disjoint_min_batch > self.batch_size:
            raise ValueError(
                "disjoint_min_batch must be <= batch_size, got "
                f"{self.disjoint_min_batch} > {self.batch_size}"
            )
        if self.disjoint_relax_sequence is not None:
            if not self.disjoint_relax_sequence:
                raise ValueError("disjoint_relax_sequence must be non-empty when provided.")
            if any(int(radius) < 0 for radius in self.disjoint_relax_sequence):
                raise ValueError("disjoint_relax_sequence entries must be non-negative.")
        if self.disjoint_fallback not in {"uniform", "best", "raise"}:
            raise ValueError(
                "disjoint_fallback must be one of {'uniform', 'best', 'raise'}, got "
                f"{self.disjoint_fallback!r}"
            )
        if not (0.0 <= self.mix_p_uniform <= 1.0):
            raise ValueError(
                f"mix_p_uniform must satisfy 0 <= p <= 1, got {self.mix_p_uniform}"
            )

        resolved_mode = self.resolved_root_sampler_mode()
        if resolved_mode != "uniform":
            if self.resolved_disjoint_restarts_k() <= 0:
                raise ValueError("resolved disjoint restarts must be positive.")
            if self.resolved_disjoint_min_batch() > self.batch_size:
                raise ValueError(
                    "resolved disjoint_min_batch must be <= batch_size, got "
                    f"{self.resolved_disjoint_min_batch()} > {self.batch_size}"
                )
        if resolved_mode == "disjoint_relax" and not self.resolved_disjoint_relax_sequence():
            raise ValueError("resolved disjoint relax sequence cannot be empty.")


@dataclass(frozen=True)
class InstanceNoiseConfig:
    """Optional discriminator-input blur (instance noise) controls."""

    enabled: bool = True
    """Enable blur noise on discriminator inputs only (default: disabled)."""

    tau_x0: float = 1.0
    """Initial X blur std in normalized units (dimensionless)."""

    tau_y0: float = 1.2
    """Initial Y blur std in normalized units (dimensionless)."""

    schedule: Literal["constant", "linear", "exp"] = "linear"
    """Annealing schedule for blur intensity over generator steps."""

    anneal_steps: int = 2000
    """Steps over which blur anneals; if 0, schedule behaves as constant."""

    min_tau: float = 0.0
    """Lower bound for tau_x/tau_y during annealing."""

    apply_to: Literal["both", "real_only"] = "both"
    """Apply blur to both real/fake batches (default) or real batches only."""

    def __post_init__(self) -> None:
        """Validate instance-noise configuration parameters."""
        if self.tau_x0 < 0.0:
            raise ValueError(f"tau_x0 must be non-negative, got {self.tau_x0}")
        if self.tau_y0 < 0.0:
            raise ValueError(f"tau_y0 must be non-negative, got {self.tau_y0}")
        if self.schedule not in ("constant", "linear", "exp"):
            raise ValueError(
                "schedule must be 'constant', 'linear', or 'exp', got "
                f"{self.schedule!r}"
            )
        if self.anneal_steps < 0:
            raise ValueError(
                f"anneal_steps must be non-negative, got {self.anneal_steps}"
            )
        if self.min_tau < 0.0:
            raise ValueError(f"min_tau must be non-negative, got {self.min_tau}")
        if self.apply_to not in ("both", "real_only"):
            raise ValueError(
                "apply_to must be 'both' or 'real_only', got "
                f"{self.apply_to!r}"
            )


@dataclass(frozen=True)
class TrueParams:
    """Ground-truth structural parameters used to create synthetic observed outcomes."""

    beta: float = 0.4
    """True peer-effect strength in the data-generating process."""

    gamma: float = 1.5
    """True coefficient on exogenous covariate X."""

    sigma_sq: float = 1.0
    """True idiosyncratic shock variance."""

    def __post_init__(self) -> None:
        """Validate parameter values."""
        if abs(self.beta) >= 1.0:
            raise ValueError(f"beta must satisfy |beta| < 1, got {self.beta}")
        if self.sigma_sq <= 0.0:
            raise ValueError(f"sigma_sq must be positive, got {self.sigma_sq}")


@dataclass(frozen=True)
class InitParams:
    """Initial parameter values for the trainable generator."""

    beta: float = 0.0
    """Initial constrained beta before optimization starts."""

    gamma: float = 0.0
    """Initial gamma before optimization starts."""

    log_sigma_sq: float = 0.0
    """Initial log-variance; `0.0` corresponds to initial variance sigma_sq = 1.0."""

    def __post_init__(self) -> None:
        """Validate that initial beta is feasible."""
        # Will be checked against beta_cap at generator construction
        pass


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for repeated Monte Carlo realizations and stopping logic."""

    n_realizations: int = 50000
    """Number of independent ground-truth realizations to run."""

    plot_every_n_realizations: int = 10
    """Regenerate Monte Carlo charts every N logged realizations during a run."""

    progress_every_n_steps: int | None = None
    """If set, print per-step parameter/loss diagnostics every N generator steps."""

    master_seed: int = 42
    """Master seed used to derive deterministic per-phase random seeds."""

    init_sigma_sq_fixed_unit: bool = True
    """If True, always initialize generator sigma_sq at 1.0 (log_sigma_sq = 0.0)."""

    init_uniform_beta_range: tuple[float, float] = (0.0, 0.5)
    """Support bounds for uniform beta initializations."""

    init_uniform_gamma_range: tuple[float, float] = (0.0, 0.5)
    """Support bounds for uniform gamma initializations."""

    init_uniform_log_sigma_sq_range: tuple[float, float] = (-0.8, 0.8)
    """Support bounds for uniform log(sigma_sq) initializations."""

    convergence_window: int = 100
    """Rolling window size for convergence diagnostics."""

    convergence_delta_d: float = 0.01
    """Absolute tolerance for discriminator rolling loss around ``2*log(2)``."""

    convergence_delta_g: float = 0.01
    """Absolute tolerance for generator rolling loss around ``log(2)``."""

    convergence_std_d_max: float = 0.1
    """Maximum allowed rolling std of discriminator loss in the convergence window."""

    convergence_std_g_max: float = 0.1
    """Maximum allowed rolling std of generator loss in the convergence window."""

    min_steps: int | None = 700
    """Optional minimum generator steps before convergence can be declared."""

    max_steps: int | None = 2000
    """Optional hard cap on generator steps per realization. ``None`` means unbounded."""

    equilibrium_dwell_steps: int = 50
    """Required consecutive in-band equilibrium steps before stopping."""

    stability_window: int = 30
    """Window length used to verify parameter stabilization before stopping."""

    stability_beta_range_tol: float = 0.01
    """Max allowed beta range over ``stability_window`` to treat beta as stable."""

    stability_gamma_range_tol: float = 0.01
    """Max allowed gamma range over ``stability_window`` to treat gamma as stable."""

    stability_sigma_sq_range_tol: float = 0.1
    """Max allowed sigma_sq range over ``stability_window`` to treat sigma_sq as stable."""

    adaptive_anneal_enabled: bool = False
    """Whether to adaptively extend annealing when convergence has not been reached."""

    adaptive_anneal_buffer_steps: int = 40
    """Extra annealing horizon maintained ahead of current step in adaptive mode."""

    output_dir: str = "artifacts/mc_asymptotic"
    """Directory where Monte Carlo outputs are written."""

    lr_g_decay_steps: tuple[int, ...] = (220, 420, 620, 780)
    """Generator learning-rate decay milestones in generator-step units."""

    lr_g_decay_factor: float = 1.0
    """Multiplicative learning-rate decay factor applied at each milestone."""

    grad_clip_norm: float = 10.0
    """Maximum norm for generator gradient clipping."""

    def __post_init__(self) -> None:
        """Validate Monte Carlo configuration parameters."""
        if self.n_realizations <= 0:
            raise ValueError(
                f"n_realizations must be positive, got {self.n_realizations}"
            )
        if self.plot_every_n_realizations <= 0:
            raise ValueError(
                "plot_every_n_realizations must be positive, got "
                f"{self.plot_every_n_realizations}"
            )
        if (
            self.progress_every_n_steps is not None
            and self.progress_every_n_steps <= 0
        ):
            raise ValueError(
                "progress_every_n_steps must be positive when provided, got "
                f"{self.progress_every_n_steps}"
            )
        if self.master_seed < 0:
            raise ValueError(f"master_seed must be non-negative, got {self.master_seed}")
        if len(self.init_uniform_beta_range) != 2:
            raise ValueError(
                "init_uniform_beta_range must contain exactly two bounds, got "
                f"{self.init_uniform_beta_range}"
            )
        if len(self.init_uniform_gamma_range) != 2:
            raise ValueError(
                "init_uniform_gamma_range must contain exactly two bounds, got "
                f"{self.init_uniform_gamma_range}"
            )
        if len(self.init_uniform_log_sigma_sq_range) != 2:
            raise ValueError(
                "init_uniform_log_sigma_sq_range must contain exactly two bounds, got "
                f"{self.init_uniform_log_sigma_sq_range}"
            )
        beta_low, beta_high = (
            float(self.init_uniform_beta_range[0]),
            float(self.init_uniform_beta_range[1]),
        )
        gamma_low, gamma_high = (
            float(self.init_uniform_gamma_range[0]),
            float(self.init_uniform_gamma_range[1]),
        )
        log_s_low, log_s_high = (
            float(self.init_uniform_log_sigma_sq_range[0]),
            float(self.init_uniform_log_sigma_sq_range[1]),
        )
        if not (beta_low < beta_high):
            raise ValueError(
                "init_uniform_beta_range must satisfy low < high, got "
                f"{self.init_uniform_beta_range}"
            )
        if beta_low <= -1.0 or beta_high >= 1.0:
            raise ValueError(
                "init_uniform_beta_range must satisfy -1 < low < high < 1 for "
                "identification (rho(W)=1 under row-stochastic normalization), got "
                f"{self.init_uniform_beta_range}"
            )
        if not (gamma_low < gamma_high):
            raise ValueError(
                "init_uniform_gamma_range must satisfy low < high, got "
                f"{self.init_uniform_gamma_range}"
            )
        if not (log_s_low < log_s_high):
            raise ValueError(
                "init_uniform_log_sigma_sq_range must satisfy low < high, got "
                f"{self.init_uniform_log_sigma_sq_range}"
            )
        if self.convergence_window <= 0:
            raise ValueError(
                f"convergence_window must be positive, got {self.convergence_window}"
            )
        if self.convergence_delta_d <= 0.0:
            raise ValueError(
                "convergence_delta_d must be positive, "
                f"got {self.convergence_delta_d}"
            )
        if self.convergence_delta_g <= 0.0:
            raise ValueError(
                "convergence_delta_g must be positive, "
                f"got {self.convergence_delta_g}"
            )
        if self.convergence_std_d_max <= 0.0:
            raise ValueError(
                "convergence_std_d_max must be positive, got "
                f"{self.convergence_std_d_max}"
            )
        if self.convergence_std_g_max <= 0.0:
            raise ValueError(
                "convergence_std_g_max must be positive, got "
                f"{self.convergence_std_g_max}"
            )
        if self.min_steps is not None and self.min_steps < 0:
            raise ValueError(f"min_steps must be non-negative when provided, got {self.min_steps}")
        if (
            self.max_steps is not None
            and self.min_steps is not None
            and self.max_steps < self.min_steps
        ):
            raise ValueError(
                f"max_steps must be >= min_steps, got {self.max_steps} < {self.min_steps}"
            )
        if self.equilibrium_dwell_steps <= 0:
            raise ValueError(
                "equilibrium_dwell_steps must be positive, got "
                f"{self.equilibrium_dwell_steps}"
            )
        if self.stability_window <= 0:
            raise ValueError(
                f"stability_window must be positive, got {self.stability_window}"
            )
        if self.max_steps is not None:
            min_start = int(self.min_steps) if self.min_steps is not None else 0
            earliest_stop = max(
                self.convergence_window + self.equilibrium_dwell_steps - 1,
                self.stability_window,
                min_start + self.equilibrium_dwell_steps - 1,
            )
            if self.max_steps < earliest_stop:
                raise ValueError(
                    "max_steps too small for convergence monitoring windows, got "
                    f"max_steps={self.max_steps}, earliest_feasible_stop={earliest_stop}"
                )
        if self.stability_beta_range_tol <= 0.0:
            raise ValueError(
                "stability_beta_range_tol must be positive, got "
                f"{self.stability_beta_range_tol}"
            )
        if self.stability_gamma_range_tol <= 0.0:
            raise ValueError(
                "stability_gamma_range_tol must be positive, got "
                f"{self.stability_gamma_range_tol}"
            )
        if self.stability_sigma_sq_range_tol <= 0.0:
            raise ValueError(
                "stability_sigma_sq_range_tol must be positive, got "
                f"{self.stability_sigma_sq_range_tol}"
            )
        if self.adaptive_anneal_buffer_steps <= 0:
            raise ValueError(
                "adaptive_anneal_buffer_steps must be positive, got "
                f"{self.adaptive_anneal_buffer_steps}"
            )
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty.")
        if any(step <= 0 for step in self.lr_g_decay_steps):
            raise ValueError("lr_g_decay_steps must contain positive step indices.")
        if tuple(sorted(self.lr_g_decay_steps)) != self.lr_g_decay_steps:
            raise ValueError("lr_g_decay_steps must be sorted in ascending order.")
        if self.lr_g_decay_factor <= 0.0 or self.lr_g_decay_factor > 1.0:
            raise ValueError(
                "lr_g_decay_factor must satisfy 0 < factor <= 1, got "
                f"{self.lr_g_decay_factor}"
            )
        if self.grad_clip_norm <= 0.0:
            raise ValueError(f"grad_clip_norm must be positive, got {self.grad_clip_norm}")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration."""

    graph: GraphConfig
    """Topology-generation controls."""

    model: ModelConfig
    """Structural-equilibrium and discriminator-capacity controls."""

    training: TrainingConfig
    """Optimization and root-sampling controls."""

    instance_noise: InstanceNoiseConfig
    """Optional blur controls for discriminator-input regularization."""

    true_params: TrueParams
    """Ground-truth parameters for synthetic data creation."""

    init_params: InitParams
    """Initial values for trainable generator parameters."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "graph": asdict(self.graph),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "instance_noise": asdict(self.instance_noise),
            "true_params": asdict(self.true_params),
            "init_params": asdict(self.init_params),
        }

    @classmethod
    def default(cls) -> ExperimentConfig:
        """Create default configuration matching design doc specification."""
        return cls(
            graph=GraphConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            instance_noise=InstanceNoiseConfig(),
            true_params=TrueParams(),
            init_params=InitParams(),
        )

    @classmethod
    def mc_default(cls) -> ExperimentConfig:
        """Create Monte Carlo-scaled configuration from the MC design document."""
        return cls(
            graph=GraphConfig(
                n_nodes=10000,
                graph_type="ba",
                ba_m=2,
                seed=42,
            ),
            model=ModelConfig(
                k=2,
                beta_cap=0.85,
                picard_tol=1e-6,
                picard_max=20,
                hidden_dim=12,
                logit_clip=4.0,
            ),
            training=TrainingConfig(
                n_steps=900,
                batch_size=17,
                n_disc=1,
                lr_d=2e-4,
                lr_g=3e-3,
                root_sampler_mode="uniform",
                root_exclusion_r=0,
                disjoint_restarts_k=1,
                disjoint_min_batch=17,
                disjoint_relax_sequence=(0,),
                disjoint_fallback="best",
                min_roots_per_call=17,
                mix_p_uniform=0.0,
            ),
            instance_noise=InstanceNoiseConfig(
                enabled=True,
                tau_x0=1.0,
                tau_y0=1.0,
                schedule="linear",
                anneal_steps=2000,
                min_tau=0.0,
                apply_to="both",
            ),
            true_params=TrueParams(
                beta=0.4,
                gamma=1.5,
                sigma_sq=1.0,
            ),
            init_params=InitParams(
                beta=0.0,
                gamma=0.0,
                log_sigma_sq=0.0,
            ),
        )


@dataclass(frozen=True)
class EffortTrueParams:
    """Ground-truth parameters for the nonlinear effort-game data generator."""

    gamma: float = 1.5
    """True exogenous covariate effect."""

    lambda_: float = 2.0 / 3.0
    """True conformity strength (implies contraction rate rho = 0.4)."""

    mu: float = 0.5
    """True precautionary motive scale."""

    r: float = 1.0
    """True precautionary curvature."""

    sigma_sq: float = 1.0
    """True idiosyncratic shock variance."""

    def __post_init__(self) -> None:
        """Validate true effort-game parameter values."""
        if self.lambda_ <= 0.0:
            raise ValueError(f"lambda_ must be positive, got {self.lambda_}")
        if self.mu < 0.0:
            raise ValueError(f"mu must be non-negative, got {self.mu}")
        if self.r <= 0.0:
            raise ValueError(f"r must be positive, got {self.r}")
        if self.sigma_sq <= 0.0:
            raise ValueError(f"sigma_sq must be positive, got {self.sigma_sq}")


@dataclass(frozen=True)
class EffortInitParams:
    """Initial parameter values for the effort-game generator."""

    gamma: float = 0.0
    """Initial gamma."""

    lambda_: float = 0.5
    """Initial constrained lambda."""

    mu: float = 0.1
    """Initial mu."""

    r: float = 1.0
    """Initial r (used only when r is learnable)."""

    log_sigma_sq: float = 0.0
    """Initial log-variance."""

    def __post_init__(self) -> None:
        """Validate effort-game initialization values."""
        if self.lambda_ <= 0.0:
            raise ValueError(f"lambda_ must be positive, got {self.lambda_}")
        if self.mu <= 0.0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.r <= 0.0:
            raise ValueError(f"r must be positive, got {self.r}")


@dataclass(frozen=True)
class EffortModelConfig:
    """Model controls for effort-game equilibrium solving and discriminator capacity."""

    k: int = 2
    """Ego radius used for rooted subgraph extraction."""

    lambda_max: float = 4.0
    """Soft upper bound for lambda via sigmoid reparameterization."""

    picard_tol: float = 1e-7
    """Picard stopping tolerance."""

    picard_max: int = 100
    """Maximum Picard iterations."""

    newton_tol: float = 1e-10
    """Newton stopping tolerance inside each Picard step."""

    newton_max: int = 8
    """Maximum Newton iterations."""

    fix_r: float | None = 1.0
    """If float, keep r fixed at this value; if None, estimate r."""

    fix_sigma_sq: float | None = 1.0
    """If float, keep sigma_sq fixed at this value; if None, estimate sigma_sq."""

    hidden_dim: int = 64
    """Discriminator hidden width."""

    discriminator_layers: int | None = None
    """Number of discriminator message-passing layers.

    When ``None``, it resolves to ``k`` so discriminator receptive field
    matches the extracted ego radius.
    """

    logit_clip: float = 10.0
    """Discriminator logit clip bound."""

    def resolved_discriminator_layers(self) -> int:
        """Resolve discriminator message-passing depth."""
        if self.discriminator_layers is None:
            return int(self.k)
        return int(self.discriminator_layers)

    def __post_init__(self) -> None:
        """Validate effort-game model configuration values."""
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.lambda_max <= 0.0:
            raise ValueError(f"lambda_max must be positive, got {self.lambda_max}")
        if self.picard_tol <= 0.0:
            raise ValueError(f"picard_tol must be positive, got {self.picard_tol}")
        if self.picard_max <= 0:
            raise ValueError(f"picard_max must be positive, got {self.picard_max}")
        if self.newton_tol <= 0.0:
            raise ValueError(f"newton_tol must be positive, got {self.newton_tol}")
        if self.newton_max <= 0:
            raise ValueError(f"newton_max must be positive, got {self.newton_max}")
        if self.fix_r is not None and self.fix_r <= 0.0:
            raise ValueError(f"fix_r must be positive when provided, got {self.fix_r}")
        if self.fix_sigma_sq is not None and self.fix_sigma_sq <= 0.0:
            raise ValueError(
                "fix_sigma_sq must be positive when provided, got "
                f"{self.fix_sigma_sq}"
            )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.discriminator_layers is not None and self.discriminator_layers <= 0:
            raise ValueError(
                "discriminator_layers must be positive when provided, got "
                f"{self.discriminator_layers}"
            )
        if self.resolved_discriminator_layers() < self.k:
            raise ValueError(
                "resolved discriminator_layers must be >= k so the discriminator can "
                f"aggregate the full {self.k}-hop ego context; got "
                f"{self.resolved_discriminator_layers()} < {self.k}"
            )
        if self.logit_clip <= 0.0:
            raise ValueError(f"logit_clip must be positive, got {self.logit_clip}")


@dataclass(frozen=True)
class EffortExperimentConfig:
    """Complete experiment configuration for nonlinear effort-game estimation."""

    graph: GraphConfig
    """Topology-generation controls."""

    model: EffortModelConfig
    """Effort-game model and discriminator controls."""

    training: TrainingConfig
    """Optimization and root-sampling controls."""

    instance_noise: InstanceNoiseConfig
    """Optional blur controls for discriminator-input regularization."""

    true_params: EffortTrueParams
    """Ground-truth effort-game parameters."""

    init_params: EffortInitParams
    """Initial generator parameters."""

    def __post_init__(self) -> None:
        """Cross-component consistency checks for effort-game experiments."""
        if self.model.resolved_discriminator_layers() < self.model.k:
            raise ValueError(
                "resolved discriminator_layers must be >= k for effort-game "
                "identification context."
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "graph": asdict(self.graph),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "instance_noise": asdict(self.instance_noise),
            "true_params": asdict(self.true_params),
            "init_params": asdict(self.init_params),
        }

    @classmethod
    def default(cls) -> EffortExperimentConfig:
        """Create default configuration for effort-game experiments."""
        return cls(
            graph=GraphConfig(),
            model=EffortModelConfig(k=2),
            training=TrainingConfig(
                n_steps=2000,
                n_disc=1,
                lr_d=2e-4,
                lr_g=7e-3,
                grad_clip_norm_g=25.0,
            ),
            instance_noise=InstanceNoiseConfig(tau_y0=1.5, anneal_steps=2500),
            true_params=EffortTrueParams(),
            init_params=EffortInitParams(),
        )
