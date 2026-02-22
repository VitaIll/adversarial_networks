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

    beta_cap: float = 0.8
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
        if not (0.0 < self.beta_cap < 1.0):
            raise ValueError(f"beta_cap must satisfy 0 < beta_cap < 1, got {self.beta_cap}")
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
