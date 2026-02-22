"""Adversarial Networks MVP: Core exports and version info."""

__version__ = "0.1.0"

# GAN components (dedicated modules for better organization)
from .generator import SCMGenerator
from .discriminator import RootedMPNNDiscriminator

# Core utilities (network and batching functions)
from .utils import (
    build_row_stochastic_W,
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
    TrueParams,
    InitParams,
)

# Constants
from . import constants

# Helper utilities
from . import io_utils
from . import visualization

__all__ = [
    # Version
    "__version__",
    # Core models
    "SCMGenerator",
    "RootedMPNNDiscriminator",
    # Core utilities
    "build_row_stochastic_W",
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
    "TrueParams",
    "InitParams",
    # Modules
    "constants",
    "io_utils",
    "visualization",
]
