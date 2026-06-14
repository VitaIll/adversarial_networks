# Architecture

## Layering

Four layers, dependencies flowing strictly downward. The split mirrors how an expert mentally models the method
(data vs. model vs. discriminator vs. estimator) and the paper's own decomposition into *computational
primitives* vs. *estimation workflow*.

```
┌─ datasets / orchestration ─────────────────────────────────────────────────┐
│  datasets.make_*           MonteCarloRunner        reporting.recovery_table  │
└──────────────────────────────┬──────────────────────────────────────────────┘
┌─ estimator (model-agnostic) ─┴──────────────────────────────────────────────┐
│  AdversarialEstimator   _run_minimax + MinimaxStepContext   StoppingRule      │
│  EstimatorConfig   EstimationResult   observability sinks                     │
└──────────────────────────────┬──────────────────────────────────────────────┘
┌─ framework objects ──────────┴──────────────────────────────────────────────┐
│  NetworkGameGenerator (base)   LinearInMeansGenerator   EffortGameGenerator   │
│  RootedMPNNDiscriminator   NetworkData   check_model   transforms             │
│  contracts: StructuralModel / TestFunction protocols                          │
└──────────────────────────────┬──────────────────────────────────────────────┘
┌─ core (fast computational primitives; pure, dependency-light) ───────────────┐
│  equilibrium.{picard, newton, solve_equilibrium}   graph   neighborhoods      │
│  objective (losses + criterion math)   ego_features   types                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**The paper names exactly three computational primitives** (Algorithm 1, §4): (i) a stable Picard equilibrium
solver, (ii) an ego-neighbourhood extraction routine, (iii) a permutation-invariant adaptive test function.
The first two are the `core/` package; the third is `RootedMPNNDiscriminator` (a provided GIN). Everything else
is estimation workflow.

## Module map

```
adversarial_networks/
  __init__.py            # curated ~24-name public surface, framework-first
  py.typed               # PEP 561 marker
  core/                  # FAST COMPUTATIONAL CORE — pure functions; numeric kernels import NO torch_geometric
    types.py             # InstanceNoiseConfigLike Protocol (typing only)
    equilibrium.py       # picard, newton (AD-diagonal or analytic Jacobian), solve_equilibrium (unroll|implicit)
    graph.py             # row_stochastic_weights, adjacency_lists_from_edge_index, normalize_adjacency
    neighborhoods.py     # precompute_balls, greedy_pack_* (numpy only)
    objective.py         # discriminator_loss, generator_(non)saturating_loss, check_gan_convergence,
                         #   instance_noise_taus, OPTIMAL_DISC_LOSS, OPTIMAL_GEN_LOSS
    ego_features.py      # extract_ego_batch  (the one core module that touches torch_geometric — primitive ii)
  generators.py          # NetworkGameGenerator (base) + LinearInMeansGenerator + EffortGameGenerator
                         #   + check_model / ModelReport
  transforms.py          # Real / Positive / Interval bijectors (declarative parameter constraints)
  discriminator.py       # RootedMPNNDiscriminator (the provided adaptive test function — primitive iii)
  contracts.py           # StructuralModel, TestFunction (extension protocols); StepMetrics, EstimationResult, MetricsObserver
  data.py                # NetworkData (domain container; mandatory outcome; owns a private EgoSubstrate)
  datasets.py            # make_linear_in_means, make_effort_game (+ _generate_graph / _build_lfr_graph)
  estimator.py           # AdversarialEstimator + _run_minimax + MinimaxStepContext + NotFittedError + warnings
  estimator_config.py    # EstimatorConfig (+ recovery_default())
  reporting.py           # recovery_table
  ego.py                 # EgoSubstrate (topology + W + X + k-ego cache + sampler; private to NetworkData)
  sampling.py            # RootSampler + RootSamplingResult + sample_roots_tensor
  losses.py              # public re-export of the three losses from core.objective
  stopping.py            # StoppingRule, StoppingDecision
  observability.py       # InMemoryHistory, ConsoleLogger, JsonlSink, CompositeObserver
  runner.py              # MonteCarloRunner, RealizationResult, derive_seed, seed_streams
  provenance.py / io_utils.py / config.py / visualization.py / plot_style.py / constants.py
```

## Dependency boundary (core ↔ workflow)

- **`core` depends only on stdlib / torch / numpy / (PyG in `ego_features` only).** It never imports a workflow
  module. This keeps the numeric kernels (`equilibrium`, `graph`, `neighborhoods`, `objective`)
  `torch_geometric`-free and independently unit-testable.
- `ego_features.extract_ego_batch` is the single, documented `core ↔ PyG` boundary (it assembles a PyG `Batch`
  — the paper's primitive ii). The pure-numeric kernels stay PyG-free.
- `OPTIMAL_DISC_LOSS`/`OPTIMAL_GEN_LOSS` are canonical in `core.objective`; `constants.py` re-imports them (no
  import cycle).
- Workflow modules import *from* core, never the reverse.

## The two abstraction families (epistemology)

The design deliberately separates **things** (objects) from **actions** (procedures), and **domain** objects
(network-econometrics vocabulary) from **ML** objects (GAN vocabulary):

| | Objects (things) | Procedures (actions) |
|---|---|---|
| **Domain** (network econometrics) | `NetworkData` (observed graph+X+Y), `EgoSubstrate` (topology), `NetworkGameGenerator`/`…Generator` (structural models) | `.simulate`, `peer_aggregate`, `best_response`/`foc_residual`, `core.equilibrium.*` (solve the equilibrium), `check_model` |
| **ML** (GAN) | `RootedMPNNDiscriminator` (adaptive test function), `EstimationResult`, `StepMetrics` | `AdversarialEstimator.fit`, `_run_minimax`, `core.objective.*` (the minimax losses) |

The **structural model is simultaneously the economic object and the GAN generator** — hence the qualified
`…Generator` names (the source method, Kaji–Manresa–Pouliot 2023, calls this object the "generator"). The
**discriminator is a nuisance** (the adaptive test function), injected like a DoubleML learner. The estimator is
**model-agnostic**: it keys off whatever parameter names `model.get_params()` exposes and never type-checks a
concrete class — only the `StructuralModel` / `TestFunction` protocols. See
[DECISIONS.md](DECISIONS.md#q2-domain-model--terminology) for the terminology rationale and peer-package
grounding.

## Why a `core/` extraction at all

CLAUDE.md forbids premature abstraction, so the cut is justified concretely: the two built-in games' Picard /
Newton loops are *structurally identical*, so `core.equilibrium.{picard, newton}` is genuinely reused — and it
is the building block a *third*, user-defined game composes without re-implementing the solver, shock draw,
autograd, or iteration counters. The extraction turns "two hand-rolled solvers" into "one reusable primitive +
three games that compose it," which is exactly what makes the package a *framework* rather than two examples.
See [EXTENDING.md](EXTENDING.md).
