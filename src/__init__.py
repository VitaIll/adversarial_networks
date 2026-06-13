"""Adversarial Networks MVP: Core exports and version info."""

__version__ = "0.1.0"

# GAN components (dedicated modules for better organization)
from .generator import SCMGenerator
from .discriminator import RootedMPNNDiscriminator

# Core utilities (network and batching functions)
from .utils import (
    build_row_stochastic_W,
    check_gan_convergence,
    compute_instance_noise_taus,
    extract_ego_batch,
)
from .root_sampling import (
    RootSampler,
    RootSamplerMode,
    DisjointFallback,
    RootSamplingResult,
    build_adjacency_from_edge_index,
    precompute_balls,
    greedy_pack_once,
    greedy_pack_best_of_k,
    sample_roots_tensor,
)

# Configuration
from .config import (
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
    InstanceNoiseConfig,
    MonteCarloConfig,
    TrueParams,
    InitParams,
)
from .io_utils import (
    append_realization_row,
    load_completed_realizations,
    save_realization_history,
)
from .visualization import (
    plot_mc_parameter_distributions,
    plot_mc_quantile_convergence_paths,
    plot_mc_quantile_loss_paths,
)

# Constants
from . import constants

# Helper utilities
from . import io_utils
from . import visualization
from .effort_generator import EffortGameGenerator
from .config import (
    EffortExperimentConfig,
    EffortModelConfig,
    EffortTrueParams,
    EffortInitParams,
)

# Estimation engine (objectives 1, 3, 6): a model-agnostic, observable estimator.
from .contracts import (
    StructuralModel,
    TestFunction,
    StepMetrics,
    EstimationResult,
    MetricsObserver,
)
from .ego import EgoSubstrate
from .estimator import AdversarialEstimator
from .estimator_config import EstimatorConfig
from .stopping import StoppingRule, StoppingDecision
from .losses import (
    discriminator_loss,
    generator_nonsaturating_loss,
    generator_saturating_loss,
)
from .observability import (
    InMemoryHistory,
    ConsoleLogger,
    JsonlSink,
    CompositeObserver,
)

__all__ = [
    # Version
    "__version__",
    # Core models
    "SCMGenerator",
    "RootedMPNNDiscriminator",
    # Core utilities
    "build_row_stochastic_W",
    "check_gan_convergence",
    "compute_instance_noise_taus",
    "extract_ego_batch",
    "RootSampler",
    "RootSamplerMode",
    "DisjointFallback",
    "RootSamplingResult",
    "build_adjacency_from_edge_index",
    "precompute_balls",
    "greedy_pack_once",
    "greedy_pack_best_of_k",
    "sample_roots_tensor",
    # Configuration
    "ExperimentConfig",
    "GraphConfig",
    "ModelConfig",
    "TrainingConfig",
    "InstanceNoiseConfig",
    "MonteCarloConfig",
    "TrueParams",
    "InitParams",
    # I/O helpers
    "append_realization_row",
    "load_completed_realizations",
    "save_realization_history",
    # Visualization helpers
    "plot_mc_parameter_distributions",
    "plot_mc_quantile_convergence_paths",
    "plot_mc_quantile_loss_paths",
    # Modules
    "constants",
    "io_utils",
    "visualization",
    "EffortGameGenerator",
    "EffortExperimentConfig",
    "EffortModelConfig",
    "EffortTrueParams",
    "EffortInitParams",
    # Estimation engine
    "AdversarialEstimator",
    "EstimatorConfig",
    "EgoSubstrate",
    "StructuralModel",
    "TestFunction",
    "StepMetrics",
    "EstimationResult",
    "MetricsObserver",
    "StoppingRule",
    "StoppingDecision",
    "discriminator_loss",
    "generator_nonsaturating_loss",
    "generator_saturating_loss",
    "InMemoryHistory",
    "ConsoleLogger",
    "JsonlSink",
    "CompositeObserver",
]
