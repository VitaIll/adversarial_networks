"""adversarial_networks — a general framework for adversarial structural estimation
of network-equilibrium models on a single graph.

The top-level surface is **framework-first** (~26 names). Advanced machinery
(``EgoSubstrate``, ``RootSampler``, the losses, ``StoppingRule``, provenance, the
experiment ``*Config`` dataclasses, the plotters, ``core.*``) is reachable from its
submodule (e.g. ``adversarial_networks.ego.EgoSubstrate``,
``adversarial_networks.losses``, ``adversarial_networks.core.equilibrium``) but is
deliberately not star-exported.

Quick start::

    import adversarial_networks as an
    data  = an.make_linear_in_means(n_nodes=10_000, graph="ba", k=2, seed=0)
    model = an.LinearInMeansGenerator(beta_cap=0.85)
    disc  = an.RootedMPNNDiscriminator(hidden_dim=12, num_layers=2, logit_clip=4.0)
    est   = an.AdversarialEstimator(model, disc,
                                    config=an.EstimatorConfig.recovery_default()).fit(data)
    est.estimates_   # coef / final / path_sd
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- the general framework ---
# Submodules kept importable for advanced use (not star-exported above).
from . import (  # noqa: F401
    config,
    constants,
    core,
    ego,
    io_utils,
    losses,
    provenance,
    sampling,
    transforms,
    visualization,
)
from .contracts import EstimationResult, StructuralModel, TestFunction
from .data import NetworkData

# --- datasets / reporting / orchestration / observability ---
from .datasets import make_effort_game, make_linear_in_means
from .discriminator import RootedMPNNDiscriminator
from .estimator import AdversarialEstimator, MinimaxStepContext, NotFittedError
from .estimator_config import EstimatorConfig
from .generators import (
    EffortGameGenerator,
    LinearInMeansGenerator,
    ModelReport,
    NetworkGameGenerator,
    check_model,
    estimate_branching,
    moment_condition_margin,
)
from .observability import CompositeObserver, ConsoleLogger, InMemoryHistory, JsonlSink
from .reporting import recovery_table
from .runner import MonteCarloRunner, RealizationResult

__all__ = [
    "__version__",
    # --- the general framework ---
    "NetworkGameGenerator",
    "StructuralModel",
    "TestFunction",
    "AdversarialEstimator",
    "EstimatorConfig",
    "EstimationResult",
    "MinimaxStepContext",
    "NotFittedError",
    "NetworkData",
    "RootedMPNNDiscriminator",
    "check_model",
    "ModelReport",
    "estimate_branching",
    "moment_condition_margin",
    "transforms",
    # --- provided model instances ---
    "LinearInMeansGenerator",
    "EffortGameGenerator",
    # --- datasets / reporting / orchestration / observability ---
    "make_linear_in_means",
    "make_effort_game",
    "recovery_table",
    "MonteCarloRunner",
    "RealizationResult",
    "InMemoryHistory",
    "ConsoleLogger",
    "JsonlSink",
    "CompositeObserver",
]
