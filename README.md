# adversarial_networks

A general framework for **adversarial structural estimation of network-equilibrium
models** on a single graph. The estimator is the adversarial minimum-distance
estimator of Illichmann & Zacchia (2026), *Adversarial Structural Estimation on
Graphs*, and its finite-moment companion note. It is a **model-agnostic tool for
any network structural model that satisfies the method's assumptions** — the two
shipped games (linear-in-means, the nonlinear effort game) are *instances* of that
class, and a user can define a new game in ~10 lines and estimate it with the same
estimator.

## The method in one paragraph

We observe a *single* graph `G = (V, E)`, node covariates `X`, and an equilibrium
outcome vector `Y` generated at an unknown structural parameter `θ₀`. For each focal
node we form its radius-`k` ego object. A **structural model** (the *generator*)
simulates equilibria `Y^θ`; an **adaptive test function** (the *discriminator*)
tries to separate observed from simulated ego objects; the **estimator** is the
alternating minimax `θ̂ ∈ argmin_θ sup_D { E_obs log D + E_θ log(1−D) }`, minimised
at `θ = θ₀` where the discriminator is at chance (`D ≡ 1/2`), so the losses sit at
`2 log 2` and `log 2` — model-free convergence diagnostics.

## Installation

```bash
pip install -e .            # editable install (adds the `adversarial_networks` package)
pip install -e .[dev]       # + pytest / ruff for development
```

Conda (exact, Windows): `conda-lock install -n adversarial_networks conda-lock.yml && conda activate adversarial_networks && pip install -e .`

Verify (from any directory):
```bash
python -c "import adversarial_networks as an; print(an.__version__)"
```

## Quick start

```python
import adversarial_networks as an

# 1. data — a provided synthetic dataset (or your own observed network via NetworkData)
data  = an.make_linear_in_means(n_nodes=10_000, graph="ba", k=2, seed=0)   # -> NetworkData

# 2. a structural model (a built-in, or YOUR NetworkGameGenerator subclass) + a discriminator
model = an.LinearInMeansGenerator(beta_cap=0.85)
disc  = an.RootedMPNNDiscriminator(hidden_dim=12, num_layers=2, logit_clip=4.0)

# 3. (optional) verify the model is admissible on this network before estimating
print(an.check_model(model, data))     # contraction (operator ∞-norm), locality, monotonicity, ...

# 4. estimate (sklearn / DoubleML-shaped)
est = an.AdversarialEstimator(model, disc, config=an.EstimatorConfig.recovery_default()).fit(data)
est.params_       # {"beta": ..., "gamma": ..., "sigma_sq": ...}
est.estimates_    # DataFrame: coef / final / path_sd  (path_sd is an optimisation diagnostic, NOT a standard error)
est.recovery_table({"beta": 0.4, "gamma": 1.5, "sigma_sq": 1.0})
```

## The admissible model class

A network game from agent preferences `U_i(y) = φ(y_i, ȳ_{N(i)}, x_i, ε_i)`. The
Nash equilibrium solves the first-order conditions; the **best response** contracts
under own-concavity + moderate social influence + a monotone shock channel, so the
equilibrium `Y^θ` is the geometric Picard fixed point.

| Game | Best response | Solver |
|---|---|---|
| Linear-in-means `Y = βWY + γX + ε` | affine, closed form | Picard |
| Effort game (nonlinear FOC) | implicit `(1+λ)z − μr e^{−rz} = b` | Picard + Newton |
| Your game | closed-form `best_response` **or** the `foc_residual` (base AD-solves it) | Picard (+ AD/analytic Newton) |

## Define your own game

Subclass `NetworkGameGenerator`, declare the parameter spaces with `transforms`, and
write only the economics — the base owns the differentiable solve, the autograd, and
the iteration bookkeeping:

```python
import torch
from adversarial_networks import NetworkGameGenerator
from adversarial_networks.transforms import Real, Positive, Interval

class LinearInMeans(NetworkGameGenerator):
    beta     = Interval(-0.85, 0.85)     # |β| < 0.85 for contraction
    gamma    = Real()
    sigma_sq = Positive()

    def best_response(self, peer_agg, X, shocks):       # peer_agg = W·Y (default mean aggregate)
        p = self.params()
        return p["beta"] * peer_agg + (p["gamma"] * X + shocks)
```

Estimate it with the **same** `AdversarialEstimator`. A full from-scratch worked
example (closed-form vs FOC routes, AD-Newton + implicit differentiation, custom
aggregates, `check_model`) is the `experiments/custom_game_model.ipynb` notebook.

## Public API

`import adversarial_networks as an` exposes a framework-first surface: the base
`NetworkGameGenerator`, the `StructuralModel`/`TestFunction` protocols,
`AdversarialEstimator` / `EstimatorConfig` / `EstimationResult` / `NotFittedError`,
`NetworkData`, `RootedMPNNDiscriminator`, `check_model` / `ModelReport`, `transforms`,
the built-in `LinearInMeansGenerator` / `EffortGameGenerator`, the `make_*` datasets,
`recovery_table`, `MonteCarloRunner`, and the observability sinks. Advanced machinery
(`ego.EgoSubstrate`, `sampling.RootSampler`, `losses`, `core.*`) lives in its
submodule.

## Status & honesty notes

- **Point estimation only.** The estimator returns point estimates and
  optimisation-convergence diagnostics. **Standard errors / confidence intervals are
  a future milestone** (a true-Fisher / GGN preconditioner via the
  `gradient_transform` seam — see `MinimaxStepContext`). `estimates_` is deliberately
  *not* an inferential `summary` and carries no `se`/`t`/`p` columns; `path_sd` is an
  optimisation-path spread, not a standard error.
- **Recovery scale.** Clean parameter recovery is an asymptotic (large-`n`) property
  of the adversarial estimator; at small fast-scale `n` the GAN dynamics are noisier.
  `EstimatorConfig.recovery_default()` is the calibrated fast-scale recipe (use it with
  the instance-noise blur shown in the notebooks); paper-scale `n` recovers tightly.

## Testing

```bash
pytest                 # the full suite
pytest -m slow         # include the (slower) recovery test
ruff check adversarial_networks tests
pytest --html=tests/reports/report.html --self-contained-html   # optional HTML report (needs the [dev] extra)
```

## References

- Illichmann, V. & Zacchia, P. (2026). *Adversarial Structural Estimation on Graphs.*
- Kaji, T., Manresa, E. & Pouliot, G. (2023). *An Adversarial Approach to Structural
  Estimation.* Econometrica 91(6): 2041-2063.
- Bramoullé, Y., Djebbari, H. & Fortin, B. (2009). *Identification of peer effects
  through social networks.* Journal of Econometrics 150(1): 41-55.
- Bai, S., Kolter, J.Z. & Koltun, V. (2019). *Deep Equilibrium Models.* (implicit
  differentiation through the fixed point.)
```
