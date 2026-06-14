# `adversarial_networks` — design docs

> Design for refactoring the package into a **general framework** for adversarial structural estimation of
> network-equilibrium models, with a comfortable `scikit-learn` / `DoubleML`-style public API, a separated fast
> computational core, and full test + notebook coverage. These documents are the reviewable design artifact;
> implementation follows the sequence in [DECISIONS.md §Implementation](DECISIONS.md#implementation-sequencing).

The package implements the adversarial minimum-distance estimator of **Illichmann & Zacchia (2026),
*Adversarial Structural Estimation on Graphs*** and its **finite-moment companion note**. The estimator is a
**general tool for *any* network structural model that satisfies the method's assumptions** — the two shipped
games (linear-in-means, the nonlinear effort game) are *instances* of that class, and a third
(saturating-peer-aggregation) is provided as a from-scratch worked example.

## The method in one paragraph

We observe a *single* graph `G = (V, E)`, node covariates `X`, and an equilibrium outcome vector `Y` generated
at an unknown structural parameter `θ₀`. For each focal node `u` we form its **ego object** `S_k(u)` — the
radius-`k` neighbourhood with covariates and outcomes. A **structural model** (the *generator*) simulates
equilibria `Y^θ`; an **adaptive test function** (the *discriminator*) tries to separate observed from simulated
ego objects; the **estimator** is the alternating minimax `θ̂ ∈ argmin_θ sup_D { E_obs log D(S) + E_θ log(1−D(S)) }`,
whose population value is `−log 4 + 2·JS(P_obs ‖ P_θ)`, minimised (to `−log 4`) exactly at `θ = θ₀`. At the
optimum the discriminator is at chance (`D ≡ 1/2`), so the discriminator loss → `2 log 2` and the structural
loss → `log 2` — model-free convergence diagnostics.

## The admissible model class (what "any conforming model" means)

A network game from agent **preferences** `U_i(y) = φ(y_i, ȳ_{N(i)}, x_i, ε_i)` (own action, a local peer
aggregate, covariate, private shock). The Nash equilibrium solves the first-order conditions `∂φ/∂y_i = 0`; the
**best response** `B_i(ȳ_{N(i)}, x_i, ε_i)` (generally implicit, Newton-solvable) **contracts** under
own-concavity (U2) + moderate social influence (U3) + a monotone shock channel (U4), so the equilibrium `Y^θ`
is the geometric **Picard fixed point**. See [THEORY_ENFORCEMENT.md](THEORY_ENFORCEMENT.md) for the full
condition list and how the package enforces / checks them.

| Game | Best response | Solver path |
|---|---|---|
| Linear-in-means `Y = βWY + γX + ε` | affine, closed form | Picard |
| Effort game (nonlinear FOC) | implicit `(1+λ)z − μr e^{−rz} = b` | Picard + Newton |
| Example 3 (saturating peer aggregation) `Y_i = φ(α + γX_i + β Σ_j a_{ij} g(Y_j)) + ε_i` | explicit, custom aggregate | Picard |

## Document map

| Doc | Contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package layout, the four layers (core / framework / estimator / datasets), the paper's three computational primitives, dependency boundaries. |
| [PUBLIC_API.md](PUBLIC_API.md) | The ~24-name public surface, every class/function signature, the sklearn/DoubleML ergonomics, learned-attribute conventions. |
| [EXTENDING.md](EXTENDING.md) | **"Define your game"** guide: the `NetworkGameGenerator` hooks, closed-form vs FOC routes, AD-Newton + implicit differentiation, declarative parameter constraints, three worked examples, `check_model`. |
| [THEORY_ENFORCEMENT.md](THEORY_ENFORCEMENT.md) | The admissible-class conditions, how `check_model` operationalises each, boundary enforcement, what is *enforced* vs *enforceable*. |
| [DECISIONS.md](DECISIONS.md) | Design decisions + rationale, the eight design-quality questions resolved, lessons borrowed from battle-tested packages, the validation-loop log, and the implementation sequence. |

## Design at a glance

```python
import adversarial_networks as an

# 1. data (a provided synthetic dataset, or your own observed network)
data  = an.make_linear_in_means(n_nodes=10_000, graph="ba", k=2, seed=0)   # -> NetworkData (y simulated)

# 2. a structural model (a built-in, or YOUR NetworkGameGenerator subclass) + a discriminator
model = an.LinearInMeansGenerator(beta_cap=0.85)
disc  = an.RootedMPNNDiscriminator(hidden_dim=12, num_layers=2, logit_clip=4.0)

# 3. estimate (sklearn/DoubleML-shaped)
est   = an.AdversarialEstimator(model, disc, config=an.EstimatorConfig.recovery_default()).fit(data)
est.params_       # {"beta": ~0.4, "gamma": ~1.5, "sigma_sq": ~1.0}
est.estimates_    # DataFrame: coef / final / path_sd   (path_sd is an optimisation diagnostic, NOT a standard error)
```

A user with a new game writes ~10 lines (a subclass with `constrained_params` + one of `best_response` /
`foc_residual`) and estimates it with the *same* `AdversarialEstimator` — see [EXTENDING.md](EXTENDING.md).

## Status & honesty notes

- **Point estimation only (for now).** The estimator returns point estimates and optimisation-convergence
  diagnostics. **Standard errors / confidence intervals are a future milestone** (a true-Fisher / GGN
  preconditioner via the `gradient_transform` seam). We adopt DoubleML/sklearn *ergonomics* (data container +
  injected model/discriminator + `fit`→`self` + a results table), not DoubleML-style inference yet. `estimates_`
  is deliberately **not** named `summary` and carries no `se`/`p`/`t` columns.
- The two built-in games are **numeric-equivalence guarded** against the current tested implementations (forward
  bit-identical; gradients `allclose(rtol=1e-5)`); see [DECISIONS.md](DECISIONS.md#numeric-equivalence).
