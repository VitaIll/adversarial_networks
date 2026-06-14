# Design decisions, Q&A resolution, and implementation plan

This document records the load-bearing design decisions and their rationale, the resolution of eight
design-quality questions, the lessons borrowed from battle-tested packages, the adversarial validation loop that
hardened the design, and the implementation sequence. It is grounded in peer-package conventions (verified
against their docs/source) and the two source papers.

## Headline decisions

| Decision | Rationale |
|---|---|
| **General framework first; the two games are instances** | The estimator is model-agnostic; the paper's admissible class is a whole family. A `NetworkGameGenerator` base + reusable `core.equilibrium` solvers turn "two hand-rolled examples" into "a framework + three games." |
| **Single `AdversarialEstimator`, no facade/engine pair** | sklearn forbids work in `__init__`; that argues for *moving* the optimiser/stopping/norm-stat build into `fit`, not wrapping one object in another. The verified loop is a free function `_run_minimax` (a frozen-mechanics seam, not a parallel class) called by both `fit` and the runner. |
| **`fit` deep-copies model/discriminator → `model_`/`discriminator_`** | Clone-safety (sklearn): the ctor objects stay pristine, `sklearn.base.clone` is genuinely fresh, re-`fit` does not warm-start. The previous design mutated the caller's modules in place. |
| **`NetworkData` has a *mandatory* outcome** | DoubleMLData-faithful (`y_col` is required). No optional-`y` two-mode object; MC substrate reuse lives in the runner, not in the public type. |
| **Config = one `EstimatorConfig` + ~4 shortcut kwargs** | DoubleML uses an object + a few scalars; flattening all ~19 knobs is the EconML 20-kwarg smell + triple-maintenance. Shortcuts (`max_steps`/`batch_size`/`lr_d`/`lr_g`/`seed`) cover the common case. |
| **`estimates_`, not `summary`; `path_sd`, not `se`; no `score()`** | The engine has no sampling-uncertainty story yet; an inferential `summary()` with a `path_sd` that *looks* like a standard error would mislead an econometrician. A raw-loss `score` would invert sklearn's greater-is-better contract. |
| **AD-Newton + implicit differentiation as the recommended route** | The per-node FOC Jacobian is diagonal, so AD gives the Newton derivative in one backward (user writes only the FOC); implicit differentiation (paper eq. 2.1) gives the structural gradient in `O(n)` memory with no second-order autograd. Default stays `unroll` (paper-faithful, tested). |
| **Rename `src` → `adversarial_networks`** | `import src` is not a public API; the distribution + repo are already `adversarial_networks`. |
| **Clean break, no deprecated aliases** | The two notebooks + experiment script + tests are all rewritten here; aliases would re-freeze the vocabulary being removed. Parameter-*key* strings are kept (a data contract). |

<a name="numeric-equivalence"></a>
## Numeric equivalence (the built-ins are guarded against drift)

Refactoring the two tested generators onto the base must not change a number. Because autograd records
*operations* (not call sites) and the shock is drawn once before the loop and reused, relocating the loop body
behind the hooks is value- and gradient-equivalent — verified by experiment:

- **`LinearInMeansGenerator`**: forward **bit-identical**; gradients **`allclose(rtol=1e-5)`** (the `gamma`
  gradient differs ~4e-6 because the base recomputes `gamma*X+shocks` inside the loop vs the current hoisted
  `base` — a benign float re-association). The baseline guard uses `rtol=1e-5`, *not* `atol=1e-6/rtol=0`.
- **`EffortGameGenerator`**: forward bit-identical; gradients `allclose ~1e-7` (the general `newton` builds two
  independent `exp(-r·z)` subgraphs).

Verified by the existing pins (Picard == dense solve @1e-5, FOC residual < 2e-4, float64 `gradcheck` @1e-4) plus
a baseline `allclose` guard comparing pre/post-refactor solver outputs on fixed seeds.

<a name="recovery-calibration"></a>
## Recovery calibration (the definition-of-done is falsifiable)

The only evidence of correct parameter recovery in the repo is at the 250k paper scale; the library defaults
differ from the working `mc_default` regime (`batch_size=17, hidden_dim=12, logit_clip=4.0, beta_cap=0.85`). So
the **first implementation step** is a calibration: run the engine at ~10k BA over a small grid, pick a config
with stable β,γ recovery over ≥3 seeds, **record the exact config + observed (β̂, γ̂, spread)**, and bake it
into `EstimatorConfig.recovery_default()` + the notebook defaults. The recovery tests assert β,γ within the
*observed* spread (not σ², which is biased at finite `n`). This gates the notebooks and is repeated (gating) for
the effort game and the custom game.

---

## The eight design-quality questions

### Q1 — Plan well-structured / covers the points?
Yes — framework-first surface, a clean core/framework/estimator/datasets layering, verification and sequencing
all present. The expert reviews found specific gaps (resolved below); the structure itself is sound.

<a name="q2-domain-model--terminology"></a>
### Q2 — Domain model & terminology optimal/recognisable; object↔procedure split correct?
Strong, verified against peer packages: `NetworkData` (≈ `DoubleMLData` / `PanelData` / PyG `Data`), `…Generator`
(Kaji–Manresa–Pouliot 2023 — the source method — *literally* call this object the "generator"), `estimates_` /
`make_*` (sklearn idioms). Object↔procedure split is clean (data vs model vs discriminator vs estimator;
simulate/fit/solve/check are procedures). **Decision:** rename `SCMGenerator` → `LinearInMeansGenerator` — a
concrete game is named for its game (like its sibling `EffortGameGenerator`); the **general
structural-causal-model abstraction is the base `NetworkGameGenerator`**, which each game specialises (peer rule:
spreg `ML_Lag`, linearmodels `PanelOLS` — instances named by model, the category on the base).

### Q3 — Boundaries enforce expectations / prevent corruption / useful warnings-errors-logs?
The in-house engine is already strong; the refactor inverts trust, so boundaries are re-established at the new
untrusted seams (user hooks, `gradient_transform`) and at the new public data path. Full table in
[THEORY_ENFORCEMENT.md §Boundaries](THEORY_ENFORCEMENT.md#boundary-enforcement-preventing-data-corruption). Also
a concrete generality bug fixed: `io_utils` hard-codes `beta/gamma/sigma_sq` (`save_realization_history` is dead
under the model-agnostic runner; `load_completed_realizations` left effort/Example-3 columns as strings) →
delete the former, coerce by column suffix in the latter.

### Q4 — Does every abstraction pull its weight?
Audited — yes. `core.equilibrium.{picard,newton}` reuse is real (the two built-ins' loops are structurally
identical). Two notes: `MinimaxStepContext.Y_obs/norm_stats` has no *current* consumer (kept, forward-looking,
to avoid a later seam-signature break when standard errors land); the four config shortcut kwargs are
two-ways-to-set-five-fields (kept; "shortcut wins" documented). We do **not** adopt sklearn's
`_parameter_constraints` machinery (the manual `__post_init__` validation is equivalent + parsimonious).
**Added** a small declarative `transforms.py` so new models declare admissible parameter spaces instead of
hand-rolling `tanh/exp/sigmoid` (+ removes the `clamp` footgun).

### Q5 / Q7 — Extensible for novel compliant models; route clear/intuitive?
The extension route is the design's strongest part — the hooks map 1:1 onto the literature's
utility→FOC→best-response→aggregate→shock pipeline and are structurally PyG's `MessagePassing` pattern. **Topology
generality (correcting a reviewer over-claim):** the model receives the *sparse* row-stochastic `W`, whose indices
are the adjacency and values are `1/degree`, so general-weight aggregates `Σ_j a_{ij} g(Y_j)` *are* expressible —
documented, with the Example-3 notebook using a genuinely non-mean-field aggregate, plus a "define your game"
guide ([EXTENDING.md](EXTENDING.md)). The contract was not changed.

### Q6 — Theoretical assumptions enforced / enforceable?
`check_model` is the enforcement surface, made sound: contraction via the **operator ∞-norm / Jacobian row-sum**
(not a median ratio), plus locality, shock monotonicity, uniqueness (multi-start), and fixed-vs-detached params,
all with a single fixed shock draw and per-check `{passed, value, threshold}`. See
[THEORY_ENFORCEMENT.md](THEORY_ENFORCEMENT.md).

<a name="q8-lessons-from-battle-tested-packages"></a>
### Q8 — Lessons from battle-tested packages (errors avoided)
| Lesson | Source precedent | Applied |
|---|---|---|
| Validate **before** mutating state | DoubleML #144 (object corrupted when a setter ran before its check) | `EgoSubstrate`/`NetworkData` validate-before-assign |
| No silent dtype/NaN coercion | sklearn `DataConversionWarning` pitfall | float32 contract *rejects*; non-finite `X`/`Y` rejected |
| Reproducible RNG | sklearn "remove `random_state=None`"; avoid global-seed mutation | explicit `torch.Generator` threaded through shock draws |
| Warnings as signal, not noise | (2000-step loops drown per-batch warnings) | warn **once** at boundaries; convergence/failure warnings |
| `fit`→`self`, trailing-underscore attrs, `check_is_fitted`/`NotFittedError` | sklearn estimator contract | followed exactly |
| Results-object discipline (no misleading SE) | statsmodels `.summary()` always carries SE; we have none | `estimates_`/`path_sd` explicitly non-inferential; no `summary`/`score` |
| Anomaly localisation for an autograd seam | PyTorch `detect_anomaly`; NumPyro valid-init guard | `gradient_transform` grad-finiteness post-check |

### Q9 — Is AD-solving the FOC the more efficient route?
Yes, when the model is smooth (the admissible class is). The per-node FOC Jacobian is diagonal, so AD gives the
Newton derivative in one backward pass and the user writes only the FOC. The genuine *efficiency* win is the
backward: implicit differentiation (`∂_θ Y = (I−A)^{-1} ∂_θ h`, the paper's eq. 2.1) is `O(n)` memory with no
second-order autograd and is exact at the fixed point — the deep-equilibrium paradigm. **Decision:** add
AD-diagonal Newton (`jacobian=None`; the `foc_residual` hook) and a `differentiation="unroll"|"implicit"`
strategy; default `"unroll"` (tested built-ins), `"implicit"` recommended for new smooth/implicit games,
cross-validated to agree on the effort game.

### User decisions
Rename to `LinearInMeansGenerator`; the topology point was confused (`W`'s sparse structure already exposes
adjacency+degree) → document + a better Example-3 aggregate, no contract change; add the `transforms` helper;
add the AD-FOC + implicit-differentiation route.

---

## Validation loop (how the design was hardened)

The design was driven to "no design mistakes" through repeated adversarial review, every finding grounded in
full source reads and read-only experiments:

| Round | Outcome |
|---|---|
| 1 (validator + challenger) | 3 blockers + 4 majors (n≈120 recovery flaky; facade mutates caller's modules; effort-kernel `eps` claim false) → single-object estimator, mandatory-`y`, config+shortcuts, `estimates_`, calibration gating |
| 2 (validator + challenger, on the general-framework pivot) | architecture confirmed sound by experiment; 3 majors → per-call Newton-counter reset, `MinimaxStepContext` seam, Example-3 gating |
| 3 (validator) | 0 blockers, 2 majors (SCM gradient is forward-bit-identical not grad-bit-identical; `check_model` must fix one shock draw) → wording + tolerance + shock-fixing |
| 4 (validator) | **CLEAN — design-complete** for technical correctness |
| 5 (3 expert reviewers: domain/terminology, boundaries/abstraction-weight, extensibility/theory) | the rename, `check_model` soundness, trust-boundary guards, `transforms`, `io_utils` fix, topology documentation, "define your game" guide, AD/implicit differentiation |

---

<a name="implementation-sequencing"></a>
## Implementation sequencing

1. **Calibrate** the fast-scale recovery config (gating — see [§Recovery calibration](#recovery-calibration)).
2. **Rename** `src` → `adversarial_networks`; add `packages.find` + `pandas` + `py.typed`; remove the duplicate
   `[tool.pytest.ini_options]`; `pip uninstall` + `install -e .`; verify `import` from a non-repo cwd + assert
   `top_level.txt`. Baseline (expect 72/1).
3. **Build `core/`** bottom-up (`types` → `objective` → `graph` → `neighborhoods` → `ego_features` →
   `equilibrium`: `picard`, `newton` with AD-diagonal/analytic Jacobian, `solve_equilibrium` unroll/implicit).
   Re-run the pins + the numeric-equivalence guard after each step.
4. **Framework**: `NetworkGameGenerator` base (hooks, `newton_solve`, counters, output guards) + `transforms.py`;
   refactor `LinearInMeansGenerator` + `EffortGameGenerator` onto the base (guarded); `check_model` + `ModelReport`.
5. **Data/estimator/orchestration**: `NetworkData` (+ `.simulate`, finite validation, validate-before-assign);
   `datasets.make_*`; `reporting.recovery_table`; `AdversarialEstimator` + `_run_minimax` + `MinimaxStepContext`
   + warnings + receptive-field guard; `observability` (store iters; explicit `Generator`); `visualization`
   (`series=`); re-point the runner (observers slot) + the `io_utils` generality fix.
6. **Surface + tests**: curate `__init__.py` (framework-first, protocols, `py.typed`); rewrite/relocate tests
   (incl. moving `src/tests/test_effort_generator.py`, fixing the stale config test); add the new public-API /
   clone-safety / core-equivalence / framework / `check_model`-soundness / AD-vs-analytic / unroll-vs-implicit
   tests.
7. **Notebooks + verify**: rewrite the two built-in notebooks (fast default + commented paper-scale switch);
   author the third (`SaturatingPeerGame` from scratch); README/pyproject; full suite green + `-m slow` + execute
   all three notebooks headless + recovery within the pinned tolerances.

## Out of scope (explicitly deferred)
- **Standard errors / confidence intervals** — the true-Fisher / GGN milestone via the `gradient_transform` seam
  (the `MinimaxStepContext` already carries what it needs); `estimates_` gains `se`/`ci` then.
- Splitting the `config.py` monolith into a subpackage (polish, no API value now).
