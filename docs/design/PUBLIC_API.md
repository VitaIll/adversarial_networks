# Public API

The top-level surface is **~24 names, framework-first**. Advanced machinery (`EgoSubstrate`, `RootSampler`,
the losses, `StoppingRule`, provenance, experiment `*Config` dataclasses, plotters, `core.*`) is reachable from
its submodule but is *not* star-exported.

```python
import adversarial_networks as an
```

```
# --- the general framework ---
NetworkGameGenerator            # abstract base: scaffold for ANY conforming network game
StructuralModel, TestFunction   # extension protocols (the estimator's true contracts)
AdversarialEstimator, EstimatorConfig, EstimationResult, NotFittedError
NetworkData                     # observed graph + covariates + outcome
RootedMPNNDiscriminator         # provided adaptive test function (bring your own TestFunction)
check_model, ModelReport        # verify a model is admissible on a network before estimating
transforms                      # Real / Positive / Interval (declarative parameter constraints)

# --- provided model instances (examples of the framework) ---
LinearInMeansGenerator          # Y = βWY + Xγ + ε
EffortGameGenerator             # nonlinear effort game (implicit FOC)

# --- datasets / reporting / orchestration / observability ---
make_linear_in_means, make_effort_game     # synthetic NetworkData (y simulated)
recovery_table
MonteCarloRunner, RealizationResult
InMemoryHistory, ConsoleLogger, JsonlSink, CompositeObserver
```

Design ergonomics follow `scikit-learn` (estimator API, `fit`→`self`, trailing-underscore learned attributes,
`NotFittedError`, `make_*` datasets) and `DoubleML` (a data container separate from the estimator + injected
"learners"); see [DECISIONS.md](DECISIONS.md#q8-lessons-from-battle-tested-packages).

---

## `NetworkData` — the data container (domain)

DoubleMLData-faithful: the **observed outcome is mandatory**. A thin owner of a private `EgoSubstrate`
(topology + `W` + `X` + the precomputed `k`-ego cache + sampler).

```python
class NetworkData:
    @classmethod
    def from_networkx(cls, graph, X, y, *, k, root_sampler_mode="uniform",
                      exclusion_r=0, disjoint_restarts_k=1, disjoint_min_batch=None,
                      disjoint_relax_sequence=(0,), disjoint_fallback="best", seed=0) -> "NetworkData"
    @classmethod
    def from_edge_index(cls, edge_index, X, y, *, k, root_sampler=None, ensure_undirected=True) -> "NetworkData"
    @classmethod
    def simulate(cls, graph_or_edge_index, X, model: StructuralModel, *, k, seed=None, **sampler_kw) -> "NetworkData"
        # build the topology, simulate the outcome from ANY model, wrap. The general "simulate on my network" path.

    @property
    def num_nodes(self) -> int
    @property
    def X(self) -> Tensor
    @property
    def y(self) -> Tensor
    @property
    def k(self) -> int
    @property
    def device(self) -> torch.device
    @property
    def topology(self) -> EgoSubstrate          # advanced: EDA / plotting needs X, edge_index, W
    def to_networkx(self) -> nx.Graph           # rebuilt from edge_index (correct length post-sanitisation; for graph plots)
```

Validation at the boundary: `X`/`y` must be finite 1-D float tensors of length `num_nodes`, on one device;
construction **validates before assigning** (no half-built object on error). The graph is sanitised
(self-loops removed, restricted to the largest connected component, relabelled) inside `EgoSubstrate`; a
`sanitization_report` is exposed.

---

## Structural models (domain ⋂ ML — the "generator")

`NetworkGameGenerator` is the abstract base; `LinearInMeansGenerator` and `EffortGameGenerator` are provided
instances. **Full extension contract in [EXTENDING.md](EXTENDING.md).** Each satisfies the `StructuralModel`
protocol (`__call__(W, X) -> Y`, `get_params`, `parameters`, `named_parameters`).

```python
class LinearInMeansGenerator(NetworkGameGenerator):     # was SCMGenerator
    def __init__(self, beta_cap=0.8, picard_tol=1e-6, picard_max=100,
                 init_beta=0.0, init_gamma=0.0, init_log_sigma_sq=0.0): ...
    def get_params(self) -> dict[str, float]            # {"beta", "gamma", "sigma_sq"}

class EffortGameGenerator(NetworkGameGenerator):
    def __init__(self, *, lambda_max=4.0, picard_tol=1e-6, picard_max=100, newton_tol=1e-10, newton_max=8,
                 fix_r=1.0, fix_sigma_sq=1.0, fixed_iterations=False,
                 init_gamma=0.0, init_lambda=0.5, init_mu=0.1, init_r=1.0, init_log_sigma_sq=0.0): ...
    def get_params(self) -> dict[str, float]            # {"gamma", "lambda_", "mu", "r", "sigma_sq"}
    @property
    def contraction_rate(self) -> float
```

The **parameter-name keys are a load-bearing data contract** (`beta`/`gamma`/`sigma_sq`/`lambda_`/`mu`/`r`/…):
they flow through `get_params` → the stopping rule → CSV `*_hat` columns → `.npz` history keys → the plotters.
They are never renamed.

---

## `RootedMPNNDiscriminator` — the adaptive test function (ML)

```python
class RootedMPNNDiscriminator(nn.Module):               # satisfies TestFunction
    def __init__(self, hidden_dim=64, num_layers=2, logit_clip=None): ...
    def forward(self, x, edge_index, root_indices) -> Tensor    # per-root logits
```

A GIN message-passing net; node features are `[X̃, Ỹ, root_marker]`. **It must cover the ego radius**: the
estimator enforces `num_layers >= data.k` at `fit` (the paper's "≥ k message-passing layers"). Bring your own
`TestFunction` by implementing `__call__(x, edge_index, root_indices) -> logits` + `parameters`.

---

## `AdversarialEstimator` — the estimator (sklearn/DoubleML-shaped)

A single object. `__init__` stores arguments verbatim (no logic); `fit` does the work and returns `self`.

```python
class AdversarialEstimator:
    def __init__(self, model: StructuralModel, discriminator: TestFunction, *,
                 config: EstimatorConfig | None = None,
                 max_steps=None, batch_size=None, lr_d=None, lr_g=None, seed=None,   # headline shortcuts (None -> config)
                 instance_noise=None, gradient_transform=None, observers=(), device=None): ...

    def fit(self, data: NetworkData) -> "AdversarialEstimator":
        # _check_data (non-NetworkData -> TypeError); receptive-field guard; deepcopy model/disc -> model_/discriminator_
        # (clone-safe; ctor objects stay pristine); run _run_minimax; warn on failure/non-convergence; return self

    # learned attributes (accessing before fit -> NotFittedError):
    #   model_, discriminator_, result_, history_, params_, params_final_,
    #   converged_, n_iter_, loss_d_, loss_g_, feature_names_

    @property
    def estimates_(self) -> pandas.DataFrame     # index=param; columns [coef, final, path_sd]; see note below
    def get_params(self, deep=True) -> dict      # sklearn introspection of __init__
    def set_params(self, **params) -> "AdversarialEstimator"
    def simulate(self, data=None, *, seed=None) -> Tensor                  # Y_sim at θ̂ (for sim-vs-obs plots)
    def discriminator_scores(self, data=None, *, n_roots=512) -> tuple[Tensor, Tensor]   # (real, fake) for the score plot
    def recovery_table(self, true_params) -> pandas.DataFrame              # convenience (== reporting.recovery_table)
```

- **`estimates_` is not an inferential `summary`.** Columns: `coef` (tail-averaged θ̂ = `result_.params`),
  `final` (last iterate), `path_sd` (std of the parameter *path* over the tail window). **`path_sd` is an
  optimisation-convergence diagnostic, NOT a standard error** — it is documented as such on the object, and there
  are deliberately no `se`/`t`/`p` columns (the estimator has no sampling-uncertainty story yet).
- **No `score()`** — a raw-loss "lower-is-better" score would invert sklearn's greater-is-better contract and
  mislead model-selection tooling.
- **Clone-safe**: `fit` deep-copies the model/discriminator, so `sklearn.base.clone(est)` is genuinely fresh and
  re-`fit` does not warm-start.
- **Warnings**: `EstimationFailedWarning` on a structural failure (non-finite simulation / NaN loss);
  a `ConvergenceWarning` analog when `not converged_` after `max_steps`. `fit` still returns `self`.

### Config: one object + a few shortcuts

`EstimatorConfig` is the single validated training-config object (max_steps, batch_size, n_disc, lr_d, lr_g,
grad_clip_norm, lr-decay schedule, the loss-band convergence criterion, parameter-stability stopping,
`differentiation="unroll"|"implicit"`, seed, …). The estimator also accepts the **handful of most-common knobs
as shortcut kwargs** (`max_steps`, `batch_size`, `lr_d`, `lr_g`, `seed`); a non-`None` shortcut overrides the
corresponding config field via `dataclasses.replace`. `EstimatorConfig.recovery_default()` returns the
**calibrated** fast-scale config that demonstrably recovers parameters (see
[DECISIONS.md](DECISIONS.md#recovery-calibration)).

---

## Datasets, reporting, orchestration

```python
def make_linear_in_means(*, n_nodes=10_000, graph="ba", k=2, true_params=..., beta_cap=0.85,
                         picard_tol=1e-6, picard_max=100, root_sampler_mode="uniform", seed=0, **graph_kwargs) -> NetworkData
def make_effort_game(*, n_nodes=10_000, graph="ba", k=2, true_params=..., lambda_max=4.0,
                     fix_r=1.0, fix_sigma_sq=1.0, ..., seed=0, **graph_kwargs) -> NetworkData
def recovery_table(estimator, true_params) -> pandas.DataFrame    # index=param; [coef, true, abs_err, path_sd]
```

`make_*` generate a graph (BA, or LFR with retry) + covariates, simulate the observed equilibrium from the
built-in model at `true_params`, and return a single `NetworkData` (the true params are an *input*, so they are
not part of a polymorphic return — there are no `return_X`/flag variants). One documented RNG lineage
(graph=`seed`, X=`seed+1`, outcome=`seed+2`) ⇒ same seed gives identical `y`.

`MonteCarloRunner` drives many independent realisations on a shared topology (resume, durable CSV/`.npz`
logging, provenance, periodic figures); it calls `_run_minimax` directly with the shared `EgoSubstrate` +
per-realisation `Y_obs` (no per-realisation topology rebuild).

---

## The future-inference seam (advanced)

`gradient_transform: Callable[[MinimaxStepContext], Mapping[str, float] | None]` runs after the structural
backward pass and before gradient clipping; it may overwrite the model's `.grad` in place and return diagnostics.
This is the documented entry point for the future true-Fisher / GGN preconditioner that will produce standard
errors. The context is a frozen dataclass carrying `step, model, discriminator, substrate, config,
instance_noise, Y_obs, norm_stats` (and `W`/`X` convenience properties) — everything that work needs, so it
will not force a later signature change. `_run_minimax` asserts gradients stay finite + shape-correct after the
transform.
