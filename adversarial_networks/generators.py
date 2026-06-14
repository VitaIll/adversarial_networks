"""Structural network-game generators: the base scaffold and the built-in games.

A *structural model* is simultaneously the economic object (a network game from
agent preferences) and the GAN *generator* (it simulates equilibria ``Y^theta``).
:class:`NetworkGameGenerator` is the abstract scaffold for **any** admissible game
(see the ``experiments/custom_game_model.ipynb`` notebook); a subclass writes only the economics —
the best response (or its FOC), and optionally the peer aggregate / shock draw /
initial state — and the base owns the differentiable equilibrium solve, the
iteration bookkeeping, input validation, and the device/dtype contracts.

Two built-in instances are provided and are **numeric-equivalence guarded** against
their original hand-rolled implementations (forward bit-identical; gradients
``allclose`` to tolerance):

* :class:`LinearInMeansGenerator` — ``Y = beta*W*Y + gamma*X + eps`` (closed form).
* :class:`EffortGameGenerator` — a nonlinear effort game whose best response solves
  an implicit FOC by Newton with an analytic diagonal Jacobian.

A third, from-scratch game (saturating peer aggregation) is the worked example in
the ``custom_game_model`` notebook.

References:
    Illichmann & Zacchia (2026), Section 2 (the admissible class) and Algorithm 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .core.equilibrium import newton, picard, solve_equilibrium
from .transforms import Transform


class NetworkGameGenerator(nn.Module):
    """Abstract base for a structural network-game generator (the GAN generator).

    Subclass hooks (fill these — the base owns everything else):

    * ``constrained_params(self) -> dict[str, Tensor]`` *(required unless you
      declare ``Transform`` fields)* — the current structural parameters in their
      admissible space.
    * ``best_response(self, peer_agg, X, shocks)`` **xor**
      ``foc_residual(self, y, peer_agg, X, shocks)`` *(exactly one)* — the best
      response in closed form, or the per-node FOC residual the base Newton-solves.
    * ``peer_aggregate(self, W, Y)`` *(optional; default row-stochastic mean
      ``W*Y``)*.
    * ``sample_shocks(self, X)`` *(optional; default scalar Gaussian from a
      ``sigma_sq`` key)*.
    * ``initial_state(self, W, X)`` *(optional; default ``zeros_like(X)``)*.

    Declarative parameters: class-level :class:`~adversarial_networks.transforms.Transform`
    attributes (``Real``/``Positive``/``Interval``) are auto-wired into learnable
    unconstrained ``nn.Parameter`` leaves, and ``constrained_params`` is assembled
    from them. The two built-ins instead hand-write their reparameterisation for
    bit-stable numerics.
    """

    _declared_transforms: dict[str, Transform] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        declared: dict[str, Transform] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Transform):
                    declared[name] = value
        cls._declared_transforms = declared

    def __init__(
        self,
        *,
        picard_tol: float = 1e-6,
        picard_max: int = 100,
        differentiation: str = "unroll",
        newton_tol: float = 1e-10,
        newton_max: int = 8,
        fixed_iterations: bool = False,
        initial_values: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not (picard_tol > 0.0):
            raise ValueError("picard_tol must be strictly positive.")
        if picard_max <= 0:
            raise ValueError("picard_max must be positive.")
        if differentiation not in {"unroll", "implicit"}:
            raise ValueError(
                f"differentiation must be 'unroll' or 'implicit', got {differentiation!r}."
            )
        if not (newton_tol > 0.0):
            raise ValueError("newton_tol must be strictly positive.")
        if newton_max <= 0:
            raise ValueError("newton_max must be positive.")

        self.picard_tol = float(picard_tol)
        self.picard_max = int(picard_max)
        self.differentiation = str(differentiation)
        self.newton_tol = float(newton_tol)
        self.newton_max = int(newton_max)
        self.fixed_iterations = bool(fixed_iterations)
        self.last_picard_iterations: int = 0
        self.last_newton_max_iters: int = 0

        # Auto-wire any declared Transform fields into learnable unconstrained leaves.
        init = dict(initial_values or {})
        for name, transform in self._declared_transforms.items():
            value = init.get(name, transform.default_constrained())
            self.register_parameter(f"_raw_{name}", nn.Parameter(transform.inverse(value)))

        # Which solve route did the subclass choose?
        self._provides_best_response = (
            type(self).best_response is not NetworkGameGenerator.best_response
        )
        self._provides_foc = type(self).foc_residual is not NetworkGameGenerator.foc_residual
        if self._provides_best_response == self._provides_foc:
            raise TypeError(
                f"{type(self).__name__} must define exactly one of best_response / foc_residual."
            )

    # ----------------------------------------------------------------- hooks
    def constrained_params(self) -> dict[str, Tensor]:
        """Return the current structural parameters in their admissible space.

        Default: apply each declared ``Transform`` to its raw leaf. Override (as the
        built-ins do) when hand-writing the reparameterisation.
        """
        if not self._declared_transforms:
            raise NotImplementedError(
                f"{type(self).__name__} must override constrained_params() or declare "
                "Transform fields."
            )
        return {
            name: transform.forward(getattr(self, f"_raw_{name}"))
            for name, transform in self._declared_transforms.items()
        }

    def params(self) -> dict[str, Tensor]:
        """Convenience alias for :meth:`constrained_params` (used inside hooks)."""
        return self.constrained_params()

    def best_response(self, peer_agg: Tensor, X: Tensor, shocks: Tensor) -> Tensor:
        """The per-node best response given the peer aggregate (closed form).

        Define this **or** :meth:`foc_residual`, not both.
        """
        raise NotImplementedError

    def foc_residual(self, y: Tensor, peer_agg: Tensor, X: Tensor, shocks: Tensor) -> Tensor:
        """The per-node first-order condition ``g_i(y_i) = 0`` (implicit best response).

        Define this **or** :meth:`best_response`, not both. The base Newton-solves it
        with an AD-computed diagonal Jacobian.
        """
        raise NotImplementedError

    def peer_aggregate(self, W: Tensor, Y: Tensor) -> Tensor:
        """The local peer aggregate. Default: the row-stochastic mean ``W * Y``.

        ``W`` is sparse with the adjacency as indices and ``1/degree`` as values, so
        a custom aggregate ``sum_j a_ij g(Y_j)`` is recoverable (see EXTENDING.md).
        """
        return torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)

    def sample_shocks(self, X: Tensor) -> Tensor:
        """Draw the reparameterised structural shocks. Default: scalar Gaussian.

        Reads a ``sigma_sq`` key from :meth:`constrained_params`; override for a
        non-Gaussian or heteroskedastic shock channel.
        """
        params = self.constrained_params()
        if "sigma_sq" not in params:
            raise KeyError(
                f"{type(self).__name__}: default sample_shocks needs a 'sigma_sq' "
                "parameter; override sample_shocks otherwise."
            )
        return torch.sqrt(params["sigma_sq"]) * torch.randn_like(X)

    def initial_state(self, W: Tensor, X: Tensor) -> Tensor:
        """The Picard start. Default: ``zeros_like(X)``."""
        return torch.zeros_like(X)

    def newton_initial_state(self, peer_agg: Tensor, X: Tensor, shocks: Tensor) -> Tensor:
        """The Newton start for the ``foc_residual`` route. Default: ``zeros_like(X)``."""
        return torch.zeros_like(X)

    # --------------------------------------------------------------- machinery
    def newton_solve(
        self,
        residual_fn,
        z0: Tensor,
        *,
        jacobian_fn=None,
    ) -> Tensor:
        """Newton-solve a per-node FOC, tracking the max iteration count.

        Helper for subclasses whose ``best_response`` is itself implicit (the effort
        game). Uses the analytic diagonal Jacobian when supplied, else AD-diagonal.
        """
        z, iters = newton(
            residual_fn,
            z0,
            tol=self.newton_tol,
            max_iter=self.newton_max,
            jacobian_fn=jacobian_fn,
            fixed_iterations=self.fixed_iterations,
        )
        self.last_newton_max_iters = max(self.last_newton_max_iters, iters)
        return z

    def _best_response_step(self, peer_agg: Tensor, X: Tensor, shocks: Tensor) -> Tensor:
        if self._provides_best_response:
            return self.best_response(peer_agg, X, shocks)
        z0 = self.newton_initial_state(peer_agg, X, shocks)
        return self.newton_solve(
            lambda z: self.foc_residual(z, peer_agg, X, shocks), z0, jacobian_fn=None
        )

    def _validate_forward_inputs(self, W: Tensor, X: Tensor) -> None:
        if not isinstance(W, Tensor):
            raise TypeError("W must be a torch.Tensor.")
        if not W.is_sparse:
            raise TypeError("W must be a sparse tensor.")
        if W.layout != torch.sparse_coo:
            raise TypeError("W must have sparse COO layout.")
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must have shape (n, n).")
        if W.dtype not in (torch.float32, torch.float64):
            raise TypeError("W must have a floating dtype.")
        if not isinstance(X, Tensor):
            raise TypeError("X must be a torch.Tensor.")
        if X.ndim != 1:
            raise ValueError("X must have shape (n,).")
        if not torch.is_floating_point(X):
            raise TypeError("X must have a floating dtype.")
        if W.shape[0] != X.shape[0]:
            raise ValueError("W and X shape mismatch: expected W.shape[0] == X.shape[0].")
        if W.device != X.device:
            raise ValueError("W and X must be on the same device.")

    def forward(self, W: Tensor, X: Tensor) -> Tensor:
        """Simulate one equilibrium outcome ``Y^theta`` with fresh shocks.

        Draws one shock vector, then solves the fixed point
        ``Y = best_response(peer_aggregate(W, Y), X, shocks)`` by the configured
        differentiation strategy. ``last_picard_iterations`` (and, for implicit-FOC
        games, ``last_newton_max_iters``) are recorded for diagnostics.
        """
        self._validate_forward_inputs(W, X)
        self.last_picard_iterations = 0
        self.last_newton_max_iters = 0

        shocks = self.sample_shocks(X)
        y0 = self.initial_state(W, X)

        def apply_step(Y: Tensor) -> Tensor:
            return self._best_response_step(self.peer_aggregate(W, Y), X, shocks)

        Y, iters = solve_equilibrium(
            apply_step,
            y0,
            tol=self.picard_tol,
            max_iter=self.picard_max,
            differentiation=self.differentiation,
            fixed_iterations=self.fixed_iterations,
        )
        self.last_picard_iterations = iters
        return Y

    def get_params(self) -> dict[str, float]:
        """Return the constrained structural parameters as detached Python floats."""
        with torch.no_grad():
            return {name: float(value.detach().reshape(-1)[0].item())
                    for name, value in self.constrained_params().items()}


class LinearInMeansGenerator(NetworkGameGenerator):
    """Linear-in-means equilibrium ``Y = beta*W*Y + gamma*X + eps``, ``eps ~ N(0, sigma^2)``.

    The peer effect is reparameterised as ``beta = beta_cap * tanh(raw_beta)`` so
    ``|beta| < beta_cap < 1`` (contraction holds for every optimiser step); the
    shock variance as ``sigma_sq = exp(log_sigma_sq) > 0``; ``gamma`` is free. The
    closed-form best response is ``beta * (W*Y) + (gamma*X + eps)`` — solved by
    Picard. (Was ``SCMGenerator``.)

    Args:
        beta_cap: Contraction cap, ``0 < beta_cap < 1``.
        picard_tol, picard_max: Picard solver controls.
        init_beta: Initial constrained ``beta`` (``|init_beta| < beta_cap``).
        init_gamma: Initial ``gamma``.
        init_log_sigma_sq: Initial ``log(sigma^2)`` (``0`` ⇒ ``sigma^2 = 1``).
        differentiation: ``"unroll"`` (default, tested) or ``"implicit"``.
    """

    def __init__(
        self,
        beta_cap: float = 0.8,
        picard_tol: float = 1e-6,
        picard_max: int = 100,
        init_beta: float = 0.0,
        init_gamma: float = 0.0,
        init_log_sigma_sq: float = 0.0,
        *,
        differentiation: str = "unroll",
    ) -> None:
        if not (0.0 < beta_cap < 1.0):
            raise ValueError("beta_cap must satisfy 0 < beta_cap < 1.")
        if abs(init_beta) >= beta_cap:
            raise ValueError("init_beta must satisfy abs(init_beta) < beta_cap.")
        super().__init__(
            picard_tol=picard_tol, picard_max=picard_max, differentiation=differentiation
        )
        self.beta_cap = float(beta_cap)
        scaled_beta = torch.tensor(init_beta / self.beta_cap, dtype=torch.float32)
        self.raw_beta = nn.Parameter(torch.atanh(scaled_beta))
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))
        self.log_sigma_sq = nn.Parameter(torch.tensor(float(init_log_sigma_sq), dtype=torch.float32))

    def constrained_params(self) -> dict[str, Tensor]:
        return {
            "beta": self.beta_cap * torch.tanh(self.raw_beta),
            "gamma": self.gamma,
            "sigma_sq": torch.exp(self.log_sigma_sq),
        }

    def sample_shocks(self, X: Tensor) -> Tensor:
        sigma = torch.sqrt(torch.exp(self.log_sigma_sq))
        return sigma * torch.randn_like(X)

    def best_response(self, peer_agg: Tensor, X: Tensor, shocks: Tensor) -> Tensor:
        beta = self.beta_cap * torch.tanh(self.raw_beta)
        # Group (gamma*X + shocks) to match the original hoisted `base` term exactly.
        return beta * peer_agg + (self.gamma * X + shocks)


class EffortGameGenerator(NetworkGameGenerator):
    """Nonlinear effort game; best response solves ``(1+lam) z - mu r e^{-r z} = b``.

    ``b = lam*(W*Y) + gamma*X + eps`` is the per-node FOC constant; the best response
    is the implicit ``z`` solving the FOC, found by Newton with the *analytic*
    diagonal Jacobian ``(1+lam) + mu r^2 e^{-r z}`` (fast, bit-stable). ``lambda``,
    ``mu``, ``r``, ``sigma^2`` are reparameterised; ``r`` and ``sigma^2`` are fixed by
    default (the finite-moment companion note's Lemma-2 regime).

    Args mirror the original generator.
    """

    def __init__(
        self,
        *,
        lambda_max: float = 4.0,
        picard_tol: float = 1e-6,
        picard_max: int = 100,
        newton_tol: float = 1e-10,
        newton_max: int = 8,
        fix_r: float | None = 1.0,
        fix_sigma_sq: float | None = 1.0,
        fixed_iterations: bool = False,
        init_gamma: float = 0.0,
        init_lambda: float = 0.5,
        init_mu: float = 0.1,
        init_r: float = 1.0,
        init_log_sigma_sq: float = 0.0,
        differentiation: str = "unroll",
    ) -> None:
        if not (lambda_max > 0.0):
            raise ValueError("lambda_max must be strictly positive.")
        if not (0.0 < init_lambda < lambda_max):
            raise ValueError("init_lambda must satisfy 0 < init_lambda < lambda_max.")
        if not (init_mu > 0.0):
            raise ValueError("init_mu must be strictly positive.")
        if fix_r is None:
            if not (init_r > 0.0):
                raise ValueError("init_r must be strictly positive when fix_r is None.")
        else:
            if not (fix_r > 0.0):
                raise ValueError("fix_r must be strictly positive when provided.")
        if fix_sigma_sq is not None and not (fix_sigma_sq > 0.0):
            raise ValueError("fix_sigma_sq must be strictly positive when provided.")
        if fix_sigma_sq is not None and float(init_log_sigma_sq) != 0.0:
            raise ValueError(
                "init_log_sigma_sq must be 0.0 when fix_sigma_sq is provided; "
                "set fix_sigma_sq=None to make sigma_sq trainable."
            )

        super().__init__(
            picard_tol=picard_tol,
            picard_max=picard_max,
            differentiation=differentiation,
            newton_tol=newton_tol,
            newton_max=newton_max,
            fixed_iterations=fixed_iterations,
        )

        self.lambda_max = float(lambda_max)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))
        scaled = torch.tensor(init_lambda / self.lambda_max, dtype=torch.float32)
        scaled = scaled.clamp(1e-6, 1.0 - 1e-6)
        self.raw_lambda = nn.Parameter(torch.logit(scaled))
        self.log_mu = nn.Parameter(torch.log(torch.tensor(float(init_mu), dtype=torch.float32)))

        if fix_r is not None:
            self._fixed_r: float | None = float(fix_r)
        else:
            self._fixed_r = None
            self.log_r = nn.Parameter(torch.log(torch.tensor(float(init_r), dtype=torch.float32)))

        if fix_sigma_sq is not None:
            self._fixed_sigma_sq: float | None = float(fix_sigma_sq)
        else:
            self._fixed_sigma_sq = None
            self.log_sigma_sq = nn.Parameter(
                torch.tensor(float(init_log_sigma_sq), dtype=torch.float32)
            )

    def _lambda(self) -> Tensor:
        return self.lambda_max * torch.sigmoid(self.raw_lambda)

    def _r(self) -> Tensor:
        if self._fixed_r is not None:
            return torch.as_tensor(self._fixed_r, dtype=self.gamma.dtype, device=self.gamma.device)
        return torch.exp(self.log_r)

    def constrained_params(self) -> dict[str, Tensor]:
        if self._fixed_sigma_sq is not None:
            sigma_sq = torch.as_tensor(
                self._fixed_sigma_sq, dtype=self.gamma.dtype, device=self.gamma.device
            )
        else:
            sigma_sq = torch.exp(self.log_sigma_sq)
        return {
            "gamma": self.gamma,
            "lambda_": self._lambda(),
            "mu": torch.exp(self.log_mu),
            "r": self._r(),
            "sigma_sq": sigma_sq,
        }

    def sample_shocks(self, X: Tensor) -> Tensor:
        if self._fixed_sigma_sq is not None:
            sigma: float | Tensor = self._fixed_sigma_sq ** 0.5
        else:
            sigma = torch.sqrt(torch.exp(self.log_sigma_sq))
        return sigma * torch.randn_like(X)

    def best_response(self, peer_agg: Tensor, X: Tensor, shocks: Tensor) -> Tensor:
        lam = self._lambda()
        mu = torch.exp(self.log_mu)
        r = self._r()
        b = lam * peer_agg + self.gamma * X + shocks

        if self._fixed_r is not None:
            z_clamp_bound: float | Tensor = 50.0 / self._fixed_r
        else:
            z_clamp_bound = 50.0 / r
        z0 = (b / (1.0 + lam)).clamp(-z_clamp_bound, z_clamp_bound)

        def residual(z: Tensor) -> Tensor:
            return (1.0 + lam) * z - mu * r * torch.exp(-r * z) - b

        def jacobian(z: Tensor) -> Tensor:
            return (1.0 + lam) + mu * r * r * torch.exp(-r * z)

        return self.newton_solve(residual, z0, jacobian_fn=jacobian)

    def get_params(self) -> dict[str, float]:
        with torch.no_grad():
            lambda_ = self._lambda()
            mu = torch.exp(self.log_mu)
            sigma_sq = (
                self._fixed_sigma_sq
                if self._fixed_sigma_sq is not None
                else float(torch.exp(self.log_sigma_sq).item())
            )
            r_val = self._fixed_r if self._fixed_r is not None else float(torch.exp(self.log_r).item())
            return {
                "gamma": float(self.gamma.item()),
                "lambda_": float(lambda_.item()),
                "mu": float(mu.item()),
                "r": float(r_val),
                "sigma_sq": float(sigma_sq),
            }

    @property
    def contraction_rate(self) -> float:
        """Current contraction rate ``rho = lambda / (1 + lambda)``."""
        with torch.no_grad():
            lambda_ = self._lambda()
            return float((lambda_ / (1.0 + lambda_)).item())


# ============================================================ admissibility check
@dataclass(frozen=True)
class CheckResult:
    """One admissibility check's outcome: ``{passed, value, threshold}`` + detail."""

    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class ModelReport:
    """The collected admissibility checks for a model on a network.

    Truthy iff every check passed. Index by check name (``report["contraction_modulus"]``)
    for the per-check :class:`CheckResult`.
    """

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def __bool__(self) -> bool:
        return self.passed

    def __iter__(self):
        return iter(self.checks)

    def __getitem__(self, name: str) -> CheckResult:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def __repr__(self) -> str:
        lines = ["ModelReport"]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  {c.name:<21} {c.value:>11.4g}  (thr {c.threshold:g})  {status}"
                         + (f"   {c.detail}" if c.detail else ""))
        lines.append(f"  => {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _extract_w_x(network) -> tuple[Tensor, Tensor]:
    """Pull ``(W, X)`` from an ``EgoSubstrate`` / ``NetworkData`` (duck-typed)."""
    W = getattr(network, "W", None)
    X = getattr(network, "X", None)
    if W is None or X is None:
        topology = getattr(network, "topology", None)
        if topology is not None:
            W = getattr(topology, "W", W)
            X = getattr(topology, "X", X)
    if W is None or X is None:
        raise TypeError("network must expose .W and .X (an EgoSubstrate or NetworkData).")
    return W, X


def _neighbor_sets(indices: Tensor, num_nodes: int) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    src = indices[0].tolist()
    dst = indices[1].tolist()
    for s, d in zip(src, dst):
        neighbors[s].add(d)
    return neighbors


def _probe_nodes(num_nodes: int, degree: Tensor, n_probe: int, seed: int) -> list[int]:
    """Probe set = highest-degree nodes (worst-case rows) ∪ a random sample.

    Including the high-degree nodes makes the operator ∞-norm robust to the star /
    public-goods counterexample whose worst row sits at the hub.
    """
    n_probe = min(n_probe, num_nodes)
    top = torch.argsort(degree, descending=True)[: (n_probe // 2) + 1].tolist()
    gen = torch.Generator().manual_seed(seed + 7)
    rand = torch.randperm(num_nodes, generator=gen)[:n_probe].tolist()
    seen: set[int] = set()
    out: list[int] = []
    for i in top + rand:
        if i not in seen:
            seen.add(i)
            out.append(int(i))
        if len(out) >= n_probe:
            break
    return out


def check_model(
    model: NetworkGameGenerator,
    network,
    *,
    n_probe: int = 256,
    locality_tol: float = 1e-4,
    uniqueness_tol: float = 1e-5,
    seed: int = 0,
) -> ModelReport:
    """Verify that ``model`` is *admissible* on ``network`` before estimating.

    Holds **one** shock draw fixed across all checks and drives the equilibrium
    through the public hooks (``best_response``/``foc_residual`` + ``peer_aggregate``),
    not ``model.forward`` (which draws its own hidden shocks). Each check reports
    ``{passed, value, threshold}``:

    * **contraction_modulus** — the operator ∞-norm ``max_i sum_j |dB_i/dY_j|`` at the
      equilibrium (Jacobian row-sum, via per-row VJP over a degree-aware probe set),
      vs threshold 1. *Not* a median ratio — a median test green-lights a
      non-contractive star public-goods map.
    * **locality_A2** — ``max |dB_i/dY_j|`` for ``j`` outside ``i``'s 1-hop.
    * **shock_monotone_U4** — ``min dB_i/d eps_i > 0``.
    * **uniqueness** — Picard from a second (random) start agrees with the first.
    * **equilibrium_residual** — ``||Y - B(Y)||_inf`` at the matched ``(Y, shocks)``.
    * **gradients** — a forward+backward reaches every learnable parameter with
      finite grads (intentionally fixed params are not learnable, so excluded).

    Args:
        model: The structural model to check.
        network: An ``EgoSubstrate`` or ``NetworkData`` exposing ``.W`` and ``.X``.
        n_probe: Number of nodes probed for the contraction/locality Jacobian rows.
        locality_tol: Tolerance for the off-neighbourhood derivative.
        uniqueness_tol: Tolerance for multi-start Picard agreement.
        seed: Seed for the fixed shock draw and the random probe/start.

    Returns:
        A :class:`ModelReport` (truthy iff all checks pass).
    """
    W, X = _extract_w_x(network)
    num_nodes = int(X.shape[0])

    torch.manual_seed(seed)
    with torch.no_grad():
        shocks = model.sample_shocks(X).detach()

    def apply_step(Y: Tensor) -> Tensor:
        return model._best_response_step(model.peer_aggregate(W, Y), X, shocks)

    with torch.no_grad():
        y_star, _ = picard(
            apply_step, model.initial_state(W, X), tol=model.picard_tol, max_iter=model.picard_max
        )
    y_star = y_star.detach()
    checks: list[CheckResult] = []

    # --- equilibrium residual (matched Y, shocks) ---
    resid_thr = max(1e-3, 100.0 * model.picard_tol)
    with torch.no_grad():
        residual = float((y_star - apply_step(y_star)).abs().max().item())
    checks.append(CheckResult("equilibrium_residual", residual < resid_thr, residual, resid_thr))

    # --- contraction (operator inf-norm) + locality over a degree-aware probe set ---
    indices = W.coalesce().indices()
    degree = torch.bincount(indices[0], minlength=num_nodes)
    neighbors = _neighbor_sets(indices, num_nodes)
    probe = _probe_nodes(num_nodes, degree, n_probe, seed)

    y_leaf = y_star.clone().requires_grad_(True)
    b_of_y = apply_step(y_leaf)
    max_row_sum = 0.0
    max_off = 0.0
    for i in probe:
        (row,) = torch.autograd.grad(b_of_y[i], y_leaf, retain_graph=True)
        max_row_sum = max(max_row_sum, float(row.abs().sum().item()))
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[i] = False
        for j in neighbors[i]:
            mask[j] = False
        if bool(mask.any()):
            max_off = max(max_off, float(row[mask].abs().max().item()))
    checks.append(CheckResult(
        "contraction_modulus", max_row_sum < 1.0, max_row_sum, 1.0,
        "operator inf-norm max_i sum_j|dB_i/dY_j|",
    ))
    checks.append(CheckResult(
        "locality_A2", max_off < locality_tol, max_off, locality_tol,
        "max |dB_i/dY_j|, j not in 1-hop(i)",
    ))

    # --- shock monotonicity dB_i/d eps_i > 0 (diagonal via one VJP) ---
    shocks_leaf = shocks.clone().requires_grad_(True)
    b_eps = model._best_response_step(model.peer_aggregate(W, y_star), X, shocks_leaf)
    (mono,) = torch.autograd.grad(b_eps.sum(), shocks_leaf)
    min_mono = float(mono.min().item())
    checks.append(CheckResult("shock_monotone_U4", min_mono > 0.0, min_mono, 0.0, "min dB_i/d eps_i"))

    # --- uniqueness: a second random start must agree ---
    uniq_thr = max(uniqueness_tol, 100.0 * model.picard_tol)
    with torch.no_grad():
        gen = torch.Generator(device=X.device).manual_seed(seed + 1)
        y_alt, _ = picard(
            apply_step,
            torch.randn(X.shape, generator=gen, device=X.device, dtype=X.dtype),
            tol=model.picard_tol,
            max_iter=model.picard_max,
        )
        uniqueness = float((y_star - y_alt).abs().max().item())
    checks.append(CheckResult("uniqueness", uniqueness < uniq_thr, uniqueness, uniq_thr,
                              "multi-start Picard agreement"))

    # --- gradient flow reaches every learnable parameter, finitely ---
    model.zero_grad(set_to_none=True)
    model(W, X).sum().backward()
    named = list(model.named_parameters())
    all_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named)
    reached = sum(1 for _, p in named if p.grad is not None and float(p.grad.abs().sum()) > 0.0)
    model.zero_grad(set_to_none=True)
    checks.append(CheckResult(
        "gradients", bool(all_finite and reached == len(named)), float(reached), float(len(named)),
        f"{reached}/{len(named)} learnable params reached; all finite={all_finite}",
    ))

    return ModelReport(checks)
