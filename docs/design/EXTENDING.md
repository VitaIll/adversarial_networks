# Define your own network game

This is the heart of the framework: a researcher who knows the network-game literature (utility → FOC → best
response → equilibrium) can implement a *new* admissible model and estimate it with the **same**
`AdversarialEstimator`, writing only the economics — never the solver, the shock draw, the autograd, or the
iteration bookkeeping.

The contract mirrors PyTorch Geometric's `MessagePassing` pattern (subclass, fill a few hooks, the base owns
the propagation). You subclass `NetworkGameGenerator` and provide:

| Hook | Required? | What it is |
|---|---|---|
| `constrained_params(self)` | **yes** | the current structural parameters, as a `dict[str, Tensor]`, in their *constrained* (admissible) space |
| `best_response(self, peer_agg, X, shocks)` **xor** `foc_residual(self, y, peer_agg, X, shocks)` | **exactly one** | the best response (closed form), **or** the per-node FOC residual (implicit; the base AD-solves it) |
| `peer_aggregate(self, W, Y)` | optional | the local aggregate (default: row-stochastic mean `W·Y`) |
| `sample_shocks(self, X)` | optional | the reparameterised shock draw (default: scalar Gaussian from a `sigma_sq` key) |
| `initial_state(self, W, X)` | optional | the Picard start (default: `zeros_like(X)`) |

The base provides `forward`, the differentiable Picard solve (optionally with an AD-Newton inner loop),
`get_params`, input validation, the iteration counters, and a `newton_solve` helper.

---

## 1. Minimal example — linear-in-means (closed form)

```python
import torch, adversarial_networks as an
from adversarial_networks.transforms import Interval, Positive, Real

class LinearInMeans(an.NetworkGameGenerator):
    beta     = Interval(-0.85, 0.85)     # declarative constraint (peer effect; |β| < 1 for contraction)
    gamma    = Real()                    # exogenous effect
    sigma_sq = Positive()                # shock variance

    def best_response(self, peer_agg, X, shocks):       # peer_agg = W·Y (the default mean aggregate)
        p = self.params()                                # {"beta", "gamma", "sigma_sq"} (constrained tensors)
        return p["beta"] * peer_agg + (p["gamma"] * X + shocks)     # explicit, additive
```

`sample_shocks` is inherited (scalar Gaussian: `sqrt(sigma_sq) · randn`). `peer_aggregate` is inherited
(`W·Y`). That is the entire model. Estimate it exactly like a built-in.

---

## 2. Implicit FOC example — the AD route (you write only the FOC)

When the best response is implicit (solves a first-order condition that has no closed form), you write the FOC
residual and the base solves it by Newton with an **AD-computed Jacobian** — you do **not** hand-derive a
derivative. This works because, within a Picard step, each node's FOC is a *scalar* equation in its own action,
so the Jacobian is diagonal and autograd returns it in one backward pass.

```python
class MyImplicitGame(an.NetworkGameGenerator):
    gamma  = Real();  lam = Interval(0, 4);  mu = Positive();  sigma_sq = Positive()

    def foc_residual(self, y, peer_agg, X, shocks):     # g_i(y_i) = ∂φ/∂y_i ;  the base Newton-solves g = 0
        p = self.params()
        b = p["lam"] * peer_agg + p["gamma"] * X + shocks
        return (1 + p["lam"]) * y - p["mu"] * torch.exp(-y) - b     # write the FOC; AD gives the derivative
```

This is the **most literature-aligned** route — a game is a utility, and its FOC is the best-response condition.
For deep nonlinear solves it pairs with the **implicit differentiation** strategy (`differentiation="implicit"`
on the config): the FOC is solved forward off-tape and the structural gradient `∂_θ Y = (I − A)^{-1} ∂_θ h`
(the paper's eq. 2.1) is obtained by an adjoint solve — `O(n)` memory, no second-order autograd, exact at the
fixed point. The default `"unroll"` strategy (autograd through the executed Picard) is what the built-ins are
tested under; the two agree to gradient tolerance.

> The **built-in `EffortGameGenerator`** uses the *closed-form* `best_response` + an *analytic*-derivative
> `newton_solve` + `"unroll"`, purely to preserve its bit-for-bit tested numerics and squeeze out the analytic
> derivative's speed. New games should prefer `foc_residual`; the AD path is cross-validated against the
> analytic one.

---

## 3. Custom aggregate example — beyond the mean field

Neither built-in overrides `peer_aggregate`, so the third worked example does, to prove the framework subsumes
general-weight / raw-neighbour aggregates — the paper's own **Example 3** `Y_i = φ(α + γX_i + β Σ_j a_{ij}
g(Y_j)) + ε_i`.

`peer_aggregate` receives the **sparse, row-stochastic `W`**. Its *indices are the adjacency* and its *values
are `1/degree`* (`degree_i = 1 / W_ij` for any neighbour `j` — only the row *sum* is normalised to 1, the
structure is fully present). So you can build **any** local aggregate from it — a degree-weighted sum, a
nonlinear `Σ_j a_{ij} g(Y_j)`, etc. — not just the mean.

```python
class SaturatingPeerGame(an.NetworkGameGenerator):
    alpha = Real();  beta = Interval(-0.6, 0.6);  gamma = Real();  sigma_sq = Positive()

    def peer_aggregate(self, W, Y):
        # raw-neighbour SUM of a saturating transform (NOT the mean): degree·(W·g(Y)) = Σ_{j∈N(i)} g(Y_j)
        degree = 1.0 / W.values()                    # recover degrees from the row-stochastic values
        return torch.sparse.mm(W, torch.tanh(Y).unsqueeze(-1)).squeeze(-1) * degree_per_row(W)

    def best_response(self, peer_agg, X, shocks):
        p = self.params()
        return torch.tanh(p["alpha"] + p["gamma"] * X + p["beta"] * peer_agg) + shocks
```

(`degree_per_row` is a one-liner over `W`'s indices; a `topology.degree` convenience is provided.) Choose `β`
strictly inside the contraction region (`beta_cap ≤ 0.6` here) so the effective modulus is `< 1` with margin —
verify with `check_model` (below).

**Multiple aggregates** (e.g. a local strategic-complement channel + a global substitute channel,
Kranton–D'Amours–Bramoullé) are expressed by returning/using several aggregates inside `best_response`
(IZ's own primitive is the single mean; the framework does not preclude more).

---

## 4. Declarative parameter constraints (`transforms`)

Instead of hand-rolling reparameterisations (`beta_cap*tanh(raw)`, `exp(log_sigma_sq)`, `lambda_max*sigmoid`)
and their inverses for initialisation — and the `clamp(1e-6, 1-1e-6)` overflow footgun — declare each
parameter's admissible space:

```python
from adversarial_networks.transforms import Real, Positive, Interval

class MyGame(an.NetworkGameGenerator):
    beta     = Interval(-1, 1)      # bijection R <-> (-1, 1)  (forward + inverse)
    sigma_sq = Positive()           # bijection R <-> (0, ∞)
    alpha    = Real()               # identity
```

Each `Transform` provides `forward` (unconstrained → constrained) and `inverse` (for initialisation from a
desired constrained value). The base wires these into a learnable unconstrained parameter per declared field and
assembles `constrained_params()` automatically. This is the GPyTorch / `torch.nn.utils.parametrize` / TFP-bijector
pattern, kept to a ~60-line dependency-light helper. You may still override `constrained_params` by hand if you
prefer.

---

## 5. Verify admissibility before estimating — `check_model`

The framework cannot *force* a user's `constrained_params` to keep the model admissible, but it can **detect**
violations loudly at a boundary instead of letting them surface as a deep `Y_sim_non_finite` failure mid-fit.

```python
report = an.check_model(my_model, network)      # holds ONE shock draw fixed across all checks
print(report)
#   contraction_modulus   ρ̂ = 0.41   (threshold < 1)   PASS     # operator ∞-norm (Jacobian row-sum), NOT a median
#   locality_A2           max |∂B_i/∂y_j|, j∉1-hop = 2e-8        PASS
#   shock_monotone_U4     min ∂B_i/∂ε_i = 0.7 > 0                PASS
#   uniqueness            multi-start Picard agree (Δ = 3e-7)    PASS
#   equilibrium_residual  ‖Y − B(Y)‖∞ = 1e-7                     PASS
#   gradients             finite, reach {alpha, beta, gamma}     PASS  (sigma_sq fixed: intentional)
```

`ModelReport` carries per-check `{passed, value, threshold}`. The **contraction check uses the operator ∞-norm
(`max_i Σ_j |∂B_i/∂y_j|`)** — the correct operationalisation of the theory's `‖B(y)−B(y')‖∞ ≤ ρ‖y−y'‖∞` *sup*
bound — not an average ratio (which can pass a non-contractive model). See
[THEORY_ENFORCEMENT.md](THEORY_ENFORCEMENT.md).

---

## 6. Estimate — identical to a built-in

```python
true_model = SaturatingPeerGame(...)                                   # at the data-generating params
data  = an.NetworkData.simulate(graph, X, model=true_model, k=2, seed=0)   # build topology + simulate + wrap
an.check_model(true_model, data)                                       # confirm admissibility
est   = an.AdversarialEstimator(SaturatingPeerGame(...),               # a fresh model at init params
                                an.RootedMPNNDiscriminator(num_layers=2),
                                config=an.EstimatorConfig.recovery_default()).fit(data)
est.params_                                                            # recovered structural parameters
```

The estimator never knew your game existed — it estimates anything satisfying the `StructuralModel` protocol.
The `custom_game_model.ipynb` notebook is the full worked template.

---

## What the base owns vs. what you own

| The base owns (you never touch) | You own (the economics) |
|---|---|
| the differentiable Picard solve + iteration counter | the best response / FOC residual |
| the AD-diagonal (or analytic) Newton + its counter + warm start | the parameters and their admissible space |
| the shock *mechanism* (reparameterised draw) | the *distribution* of shocks (if non-default) |
| `forward`, `get_params`, input validation, device/dtype contracts | the peer aggregate (if non-default) |
| `unroll` vs `implicit` structural-gradient strategy | nothing about gradients |

If even the base is too opinionated (a closed-form equilibrium, an Anderson-accelerated solve, a multi-dimensional
action), implement the bare `StructuralModel` protocol directly — the estimator only ever requires the protocol.
