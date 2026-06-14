# Theory enforcement & boundaries

How the package keeps a model *admissible* (satisfying the method's assumptions) and prevents silent data
corruption — distinguishing what is **enforced** (cannot be violated) from what is **checkable / detected**
(violations surface loudly at a boundary) from what is **documented** (the user's responsibility).

## The admissible-class conditions

From Illichmann–Zacchia (2026) §2 and the finite-moment note §2–3:

| Code | Condition | Operational meaning |
|---|---|---|
| A1 | Bounded degree | finite ego-ball sizes (uniformly) |
| A2 | Locality | coordinate `i` of the equilibrium map depends only on the 1-hop neighbourhood of `i` |
| B1 / U2+U3 | Contraction | the best response contracts: `‖B(y)−B(y')‖∞ ≤ ρ‖y−y'‖∞`, `ρ < 1` (own-concavity + moderate social influence) ⇒ unique equilibrium + geometric Picard |
| B2 / U4 | Shock channel | additive shocks (B2), or more generally a monotone shock channel (U4: `∂B_i/∂ε_i > 0`) |
| B3 | Smooth i.i.d. shocks | `ε` i.i.d. with a `C¹` density (identification regularity) |
| D1 / U1 | Smoothness | `φ` (hence the equilibrium map) is `C²` ⇒ autograd through the unrolled / implicit solve is valid |
| M | Moment condition | `ρ^p λ < 1` couples contraction `ρ`, graph branching `λ`, shock tail `p` (a consistency-rate condition, not a per-model check) |

## Enforced (cannot be violated)

| What | Where | How |
|---|---|---|
| Locality (A2) for the **default** aggregate | `peer_aggregate = W·Y` | structural: the row-stochastic `W` is sparse along graph edges, so `B_i` depends only on 1-hop neighbours by construction |
| Smoothness used by autograd (D1) | the solve | the equilibrium is built from differentiable torch ops; non-differentiable user code simply fails to produce gradients (caught — see boundaries) |
| Discriminator covers the ego radius | `AdversarialEstimator.fit` | hard `ValueError` if `num_layers < k` (the paper's "≥ k message-passing layers") |
| Parameter admissibility *space* | `transforms` | a declared `Interval(-1,1)`/`Positive()` is a bijection from `ℝ`, so the constrained value is *always* in range — no optimiser step can leave it |

## Checkable / detected (`check_model`)

`check_model(model, network)` is the admissibility surface. It **holds one shock draw fixed** across all checks
(the contraction/residual checks are only well-defined for a fixed shock realisation) and drives the equilibrium
through the public hooks (`best_response`/`foc_residual` + `peer_aggregate`), *not* `model.forward` (which draws
its own hidden shocks). Each check reports `{passed, value, threshold}`:

| Check | Theory | Correct operationalisation |
|---|---|---|
| **Contraction (B1/U2+U3)** | `‖B(y)−B(y')‖∞ ≤ ρ‖y−y'‖∞`, a **sup over directions** | `ρ̂ = max_i Σ_j \|∂B_i/∂y_j\|` (the Jacobian operator ∞-norm / row-sum at the equilibrium, via autograd VJP or Jacobian power-iteration) vs threshold 1. **Not a median ratio** — a median/random-direction ratio underestimates the sup and can green-light a non-contractive, non-unique model (a star public-goods map passes a median test at 0.37 while its true ‖A‖∞ ≈ 10). |
| **Locality (A2)** | `B_i` depends only on 1-hop(`i`) | probe `∂B_i/∂y_j ≈ 0` for `j ∉ 1-hop(i)` on sampled `i` (a custom `peer_aggregate` may break this; the default cannot) |
| **Shock monotonicity (U4)** | `∂B_i/∂ε_i ∈ [τ, τ̄] > 0` | VJP of `B` w.r.t. the fixed shocks; require `> 0` |
| **Uniqueness** | unique fixed point, not merely convergent-from-zero | Picard from ≥ 2 distinct starts (`initial_state` + a random start) converges to the same `Y` within tol |
| **Equilibrium residual** | `Y` is a fixed point | `‖Y − B(peer_aggregate(W,Y), X, shocks)‖∞` small, with matched `(Y, shocks)` |
| **Gradient flow** | `θ` is identified through the solve | a forward+backward reaches every learnable parameter with finite grads; **intentionally fixed** params (detached / `torch.as_tensor`, e.g. effort `fix_r`/`fix_sigma_sq`) are distinguished from **accidentally detached** ones |

The Example-3 notebook calls `check_model` before estimating; the test suite includes a deliberately
non-contractive model that `check_model` must **fail**.

## Documented (the user's responsibility)

- Choosing parameters inside the contraction region (the base cannot force a user's `constrained_params` to be
  contractive — but `check_model` detects when they are not).
- The B2-additive vs. monotone-FOC distinction: the shipped effort game's shock is additive only in the *FOC
  constant*, not in the outcome (finite-moment Lemma 2, not the B2 consistency theorem). A per-model
  **`is_additive` flag** is recorded so the future Fisher / standard-error path (which needs the additive
  change-of-variables) knows when it applies.
- The moment condition M (`ρ^p λ < 1`) is a property of the graph *ensemble* and shock tail, relevant to the
  consistency *rate*, not a per-fit check.

---

## Boundary enforcement (preventing data corruption)

The in-house engine already validated heavily (typed `runtime_checkable` protocols, attributable
construction-time `TypeError`/`ValueError`, isolated observer dispatch, preserved non-finite failure paths). The
refactor **inverts trust** — it now runs user-supplied hooks and a user `gradient_transform` in the hot path — so
the boundaries are re-established at those new seams:

| Seam | Guard |
|---|---|
| **User hooks** (`best_response`/`foc_residual`/`peer_aggregate`/`sample_shocks`) | a one-time pre-loop post-condition check that the output is a finite, grad-connected float tensor of shape `X.shape` on `X.device` — raised with the hook + model name (not a cryptic failure many Picard steps later) |
| **`gradient_transform`** | after the transform, assert grads are finite and shapes unchanged, *before* clipping (anomaly localisation, à la `torch.detect_anomaly` / NumPyro's valid-init guard) |
| **`NetworkData` construction** | reject non-finite `X`/`Y` (`torch.isfinite`, the DoubleMLData `force_all_x_finite` idiom); **validate before assigning** any attribute (no half-built object on a validation error — the DoubleML #144 anti-pattern) |
| **`fit` outcome** | `EstimationFailedWarning` on a structural failure; a `ConvergenceWarning` analog when unconverged after `max_steps`; never a silent all-NaN `params_` |
| **dtype/device** | the float32 contract *rejects* rather than silently downcasts (the opposite of a silent `DataConversionWarning`); gradchecks run in float64 with no silent crossing |
| **RNG** | shock draws thread an explicit `torch.Generator` rather than mutating the global torch RNG, so reproducibility is a guarantee and a shared topology is genuinely fit-safe |

## Diagnostics surfaced (interpretable run logs)

Beyond the per-step `StepMetrics` / `InMemoryHistory` / `ConsoleLogger` / `JsonlSink` and the run-level
provenance manifest, the following are surfaced as **actionable warnings / `EstimationResult.extras`** rather
than buried: Picard / Newton hitting the iteration cap (a non-converged equilibrium silently feeding the
discriminator), sampler shortfall (`RootSamplingResult.met_target` / `fallback_reason`), and a
weak-identification flag. Boundary-condition warnings (e.g. a non-default `instance_noise.apply_to`) are emitted
**once** at the `fit`/config boundary, not on every minibatch.
