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

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .core.equilibrium import (
    EquilibriumNotConverged,
    newton,
    picard,
    solve_equilibrium,
)
from .core.graph import adjacency_lists_from_edge_index
from .core.neighborhoods import precompute_balls
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
    * ``sample_shocks(self, X)`` *(optional; default per-node ``(n,)`` scalar Gaussian
      from a ``sigma_sq`` key)*.
    * ``initial_state(self, W, X)`` *(optional; default per-node ``(n,)`` zeros)*.

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
        picard_max: int = 200,
        differentiation: str = "unroll",
        newton_tol: float = 1e-10,
        newton_max: int = 8,
        fixed_iterations: bool = False,
        raise_on_nonconvergence: bool = False,
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
        # Base cap raised from 100 to 200: high-contraction models (rho near 1) need a
        # larger budget (paper T = O(log(1/tol)/|log rho|); rho=0.9 already needs ~131
        # iterations at tol=1e-6). Under-solving is now observable via
        # last_picard_residual/last_picard_converged and the estimator's cap warning.
        self.picard_max = int(picard_max)
        self.differentiation = str(differentiation)
        self.newton_tol = float(newton_tol)
        self.newton_max = int(newton_max)
        self.fixed_iterations = bool(fixed_iterations)
        self.raise_on_nonconvergence = bool(raise_on_nonconvergence)
        self.last_picard_iterations: int = 0
        self.last_newton_max_iters: int = 0
        # Convergence observability (distinct from the GAN loss-band in_equilibrium):
        # the final solver residual and whether the tolerance test actually fired.
        self.last_picard_residual: float = 0.0
        self.last_picard_converged: bool = True
        self.last_newton_residual: float = 0.0
        self.last_newton_converged: bool = True

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
        """The local peer aggregate. Default: the row-stochastic mean ``W @ Y``.

        ``W`` is a coalesced sparse-COO matrix that carries the local *topology*: its
        ``indices()`` are the ``(2, num_edges)`` ``(i, j)`` edge list (the binary
        adjacency) and each stored value is ``1/deg_i`` (so the row sums are 1). The
        per-edge weights ``a_ij`` of a general aggregate ``sum_j a_ij g(Y_j)`` (the
        paper's Example 3) are **user-supplied** — they are not encoded in ``W`` — but
        an override can compute that aggregate from ``W``'s edge list plus the user's
        own ``a_ij``, so it is not limited to the mean.

        * **Raw neighbour sum** ``sum_{j in N(i)} g(Y_j)``. Since ``(W @ g(Y))_i`` is
          the *mean* ``(1/deg_i) sum_j g(Y_j)``, multiply back by the per-row degree
          (recovered by counting edges per row)::

            n = Y.shape[0]
            deg = torch.bincount(W.coalesce().indices()[0], minlength=n).to(Y.dtype)
            mean_g = torch.sparse.mm(W, g(Y).unsqueeze(-1)).squeeze(-1)
            return mean_g * deg                          # = sum_{j in N(i)} g(Y_j)

        * **General per-neighbour weights** ``sum_j a_ij g(Y_j)`` with arbitrary,
          user-supplied ``a_ij`` (varying across a row, not a function of degree).
          Read the edge list, gather ``Y_j``, apply ``g`` and your ``a_ij``, and
          scatter-sum into the rows::

            i, j = W.coalesce().indices()                # row i, neighbour j
            contrib = a_ij * g(Y[j])                     # a_ij is yours, indexed per-edge
            return torch.zeros_like(Y).scatter_add_(0, i, contrib)

        See ``docs/design/EXTENDING.md`` (section 3) for the full worked example.
        """
        return torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)

    def sample_shocks(self, X: Tensor) -> Tensor:
        """Draw the reparameterised per-node structural shocks. Default: scalar Gaussian.

        Reads a ``sigma_sq`` key from :meth:`constrained_params`; override for a
        non-Gaussian or heteroskedastic shock channel. The shock is per node, shape
        ``(n,)``, regardless of the covariate width ``d_x`` (for a 1-D ``X`` this is
        bit-identical to ``randn_like(X)`` — same shape, dtype, device, and RNG draws).
        """
        params = self.constrained_params()
        if "sigma_sq" not in params:
            raise KeyError(
                f"{type(self).__name__}: default sample_shocks needs a 'sigma_sq' "
                "parameter; override sample_shocks otherwise."
            )
        return torch.sqrt(params["sigma_sq"]) * torch.randn(
            X.shape[0], dtype=X.dtype, device=X.device
        )

    def initial_state(self, W: Tensor, X: Tensor) -> Tensor:
        """The Picard start. Default: per-node ``zeros`` of shape ``(n,)``.

        For a 1-D ``X`` this is bit-identical to ``zeros_like(X)``.
        """
        return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

    def newton_initial_state(self, peer_agg: Tensor, X: Tensor, shocks: Tensor) -> Tensor:
        """The Newton start for the ``foc_residual`` route. Default: per-node ``zeros``.

        Shape ``(n,)``; for a 1-D ``X`` this is bit-identical to ``zeros_like(X)``.
        """
        return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

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

        Newton runs once per Picard step; ``last_newton_max_iters`` keeps the worst-case
        iteration count and ``last_newton_residual``/``last_newton_converged`` keep the
        worst-case (largest residual / any-non-converged) outcome across the Picard
        sweep, so a single under-solved inner step is observable.
        """
        result = newton(
            residual_fn,
            z0,
            tol=self.newton_tol,
            max_iter=self.newton_max,
            jacobian_fn=jacobian_fn,
            fixed_iterations=self.fixed_iterations,
        )
        self.last_newton_max_iters = max(self.last_newton_max_iters, result.iters)
        self.last_newton_residual = max(self.last_newton_residual, result.residual)
        self.last_newton_converged = self.last_newton_converged and result.converged
        return result.Y

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
        if X.ndim not in (1, 2):
            raise ValueError("X must have shape (n,) or (n, d_x).")
        if X.ndim == 2 and X.shape[1] < 1:
            raise ValueError("X must have at least one covariate column (d_x >= 1).")
        if not torch.is_floating_point(X):
            raise TypeError("X must have a floating dtype.")
        if W.shape[0] != X.shape[0]:
            raise ValueError("W and X shape mismatch: expected W.shape[0] == X.shape[0].")
        if W.device != X.device:
            raise ValueError("W and X must be on the same device.")

    def _check_hook_output(
        self, value: object, hook: str, *, n: int, device: torch.device, check_grad: bool
    ) -> None:
        """Post-condition on a user hook's output: finite floating ``(n,)`` on ``device``.

        Used by :meth:`forward` to fail loudly *at the hook* — naming the hook and the
        model class — rather than many Picard steps later, when a malformed
        ``peer_aggregate`` / ``best_response`` / ``foc_residual`` output corrupts the
        solve. ``check_grad`` additionally requires the output to be grad-connected (the
        structural gradient cannot flow otherwise).
        """
        cls = type(self).__name__
        if not isinstance(value, Tensor):
            raise ValueError(
                f"{cls}.{hook} must return a torch.Tensor, got {type(value).__name__}."
            )
        if not torch.is_floating_point(value):
            raise ValueError(f"{cls}.{hook} must return a floating tensor, got dtype {value.dtype}.")
        if value.shape != (n,):
            raise ValueError(
                f"{cls}.{hook} must return a per-node tensor of shape ({n},), got "
                f"{tuple(value.shape)}."
            )
        if value.device != device:
            raise ValueError(
                f"{cls}.{hook} must return a tensor on {device}, got {value.device}."
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{cls}.{hook} returned non-finite values (NaN/inf).")
        if check_grad and not value.requires_grad:
            raise ValueError(
                f"{cls}.{hook} output is not grad-connected (requires_grad is False) while "
                "a model parameter requires grad; the structural gradient cannot flow through "
                "the equilibrium solve."
            )

    def forward(self, W: Tensor, X: Tensor) -> Tensor:
        """Simulate one equilibrium outcome ``Y^theta`` with fresh shocks.

        Draws one shock vector, then solves the fixed point
        ``Y = best_response(peer_aggregate(W, Y), X, shocks)`` by the configured
        differentiation strategy. ``last_picard_iterations``/``last_picard_residual``/
        ``last_picard_converged`` (and, for implicit-FOC / Newton best-response games,
        the ``last_newton_*`` equivalents) are recorded for diagnostics.

        When ``raise_on_nonconvergence`` is set and the Picard solve hit its cap without
        the tolerance test firing, raises :class:`EquilibriumNotConverged` with the
        attributable residual/tol/cap (skipped under ``fixed_iterations``, where no
        convergence test runs by design).
        """
        self._validate_forward_inputs(W, X)
        self.last_picard_iterations = 0
        self.last_newton_max_iters = 0
        self.last_picard_residual = 0.0
        self.last_picard_converged = True
        self.last_newton_residual = 0.0
        self.last_newton_converged = True

        shocks = self.sample_shocks(X)
        y0 = self.initial_state(W, X)

        # One-time boundary post-condition on the user hooks: validate the peer-aggregate
        # and best-response outputs on the FIRST apply_step only (the flag short-circuits
        # every later call, so the solver loop carries zero overhead). This reuses the
        # hooks' existing call sites — no extra hook invocation, no shock draw, and no
        # mutation of the iteration counters — so the built-ins stay bit-identical.
        n = X.shape[0]
        validated = [False]

        def apply_step(Y: Tensor) -> Tensor:
            peer_agg = self.peer_aggregate(W, Y)
            if not validated[0]:
                self._check_hook_output(
                    peer_agg, "peer_aggregate", n=n, device=X.device, check_grad=False
                )
            br = self._best_response_step(peer_agg, X, shocks)
            if not validated[0]:
                hook = "best_response" if self._provides_best_response else "foc_residual"
                check_grad = torch.is_grad_enabled() and any(
                    p.requires_grad for p in self.parameters()
                )
                self._check_hook_output(br, hook, n=n, device=X.device, check_grad=check_grad)
                validated[0] = True
            return br

        result = solve_equilibrium(
            apply_step,
            y0,
            tol=self.picard_tol,
            max_iter=self.picard_max,
            differentiation=self.differentiation,
            fixed_iterations=self.fixed_iterations,
        )
        self.last_picard_iterations = result.iters
        self.last_picard_residual = result.residual
        # fixed_iterations runs no convergence test (gradcheck path); report it as
        # converged so it neither false-flags nor trips raise_on_nonconvergence.
        self.last_picard_converged = result.converged or self.fixed_iterations
        if (
            self.raise_on_nonconvergence
            and not self.fixed_iterations
            and not result.converged
        ):
            raise EquilibriumNotConverged(
                residual=result.residual, tol=self.picard_tol, max_iter=self.picard_max
            )
        return result.Y

    def get_params(self) -> dict[str, float]:
        """Return the constrained structural parameters as detached Python floats.

        A scalar parameter (``numel == 1``) maps to ``{name: float}``. A vector
        parameter (``numel > 1``) — e.g. a ``(d_x,)`` covariate-effect ``gamma`` — is
        expanded into indexed keys ``{f"{name}[{i}]": float}`` (one per component) so
        no component is silently dropped.
        """
        out: dict[str, float] = {}
        with torch.no_grad():
            for name, value in self.constrained_params().items():
                flat = value.detach().reshape(-1)
                if int(flat.numel()) == 1:
                    out[name] = float(flat[0].item())
                else:
                    for i in range(int(flat.numel())):
                        out[f"{name}[{i}]"] = float(flat[i].item())
        return out


def _require_scalar_covariate(cls_name: str, X: Tensor) -> None:
    """Reject a vector covariate for a scalar-only built-in, attributably.

    The base validator widened ``X`` to accept ``(n,)`` or ``(n, d_x)`` for the general
    framework path, but the built-ins are intentionally scalar (their ``best_response``
    computes ``gamma * X`` for a scalar ``gamma``). A ``(n, d_x>=2)`` ``X`` would otherwise
    crash with a raw broadcast ``RuntimeError`` deep inside ``best_response`` (before the
    ``forward`` post-condition can attribute it). Fail at the boundary instead.
    """
    if X.ndim != 1:
        raise ValueError(
            f"{cls_name} supports a scalar covariate only (X shape (n,)); got X with "
            f"shape {tuple(X.shape)}. Write a custom NetworkGameGenerator with a "
            "vector-gamma best_response for d_x > 1 covariates."
        )


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
        picard_max: int = 200,
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

    def _validate_forward_inputs(self, W: Tensor, X: Tensor) -> None:
        super()._validate_forward_inputs(W, X)
        _require_scalar_covariate(type(self).__name__, X)

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
        picard_max: int = 200,
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

    def _validate_forward_inputs(self, W: Tensor, X: Tensor) -> None:
        super()._validate_forward_inputs(W, X)
        _require_scalar_covariate(type(self).__name__, X)

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


def _off_ball_masks(
    indices: Tensor,
    neighbors: list[set[int]],
    probe: list[int],
    num_nodes: int,
    interaction_radius: int,
) -> dict[int, set[int]]:
    """The *in-ball* node set ``B_{r0}(i)`` (to exclude) for each probe node ``i``.

    The locality (A2) check probes ``∂B_i/∂Y_j`` for ``j`` *outside* node ``i``'s
    interaction ball ``B_{r0}(i)`` (all nodes within ``interaction_radius`` hops,
    inclusive). For the default ``interaction_radius == 1`` the ball is exactly
    ``{i} ∪ 1-hop(i)`` — this branch returns precisely the set the original 1-hop mask
    excluded, so the ``r0 = 1`` path is bit-identical. For ``r0 >= 2`` the balls are
    materialised by BFS-to-radius via :func:`~adversarial_networks.core.neighborhoods.precompute_balls`
    over the graph adjacency (the coalesced ``W`` indices *are* the adjacency).
    """
    if interaction_radius == 1:
        # Bit-identical to the original 1-hop mask: B_1(i) = {i} ∪ neighbors(i).
        return {i: ({i} | neighbors[i]) for i in probe}
    edge_index = indices.to(torch.long)
    adjacency = adjacency_lists_from_edge_index(edge_index, num_nodes)
    balls = precompute_balls(adjacency, [interaction_radius])[interaction_radius]
    return {i: {int(v) for v in balls[i].tolist()} for i in probe}


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
    interaction_radius: int = 1,
    seed: int = 0,
) -> ModelReport:
    """Verify that ``model`` is *admissible* on ``network`` before estimating.

    Holds **one** shock draw fixed across all checks and drives the equilibrium
    through the public hooks (``best_response``/``foc_residual`` + ``peer_aggregate``),
    not ``model.forward`` (which draws its own hidden shocks). Works for a scalar
    (``X`` of shape ``(n,)``) or vector (``(n, d_x)``) covariate: every outcome-shaped
    tensor stays ``(n,)`` regardless of ``d_x``. Each check reports
    ``{passed, value, threshold}``:

    * **contraction_modulus** — the operator ∞-norm ``max_i sum_j |dB_i/dY_j|`` at the
      equilibrium (Jacobian row-sum, via per-row VJP over a degree-aware probe set),
      vs threshold 1. *Not* a median ratio — a median test green-lights a
      non-contractive star public-goods map. Its detail string flags a ``DEGENERATE``
      probe (negligible peer interaction at the current params), where contraction and
      locality pass trivially and certify nothing.
    * **locality_A2** — ``max |dB_i/dY_j|`` for ``j`` outside ``i``'s interaction ball
      ``B_{r0}(i)`` (all nodes within ``interaction_radius`` hops). With the default
      ``interaction_radius == 1`` this is the 1-hop off-neighbourhood derivative; a
      model whose ``peer_aggregate`` legitimately reaches radius ``r0 >= 1`` (paper fn. 5)
      is checked at that radius.
    * **shock_monotone_U4** — ``min dB_i/d eps_i > 0``.
    * **uniqueness** — Picard from a second (random) start agrees with the first.
    * **equilibrium_residual** — ``||Y - B(Y)||_inf`` at the matched ``(Y, shocks)``.
    * **gradients** — a forward+backward reaches every learnable parameter with
      finite grads (intentionally fixed params are not learnable, so excluded).

    This is a per-fit admissibility gate. The finite-moment binding condition (M)
    ``rho_bar^p * lambda < 1`` and the consistency rate ``gamma`` — properties of the
    *graph ensemble* and shock tail, not of one fit — are surfaced separately by the
    module-level :func:`estimate_branching` / :func:`moment_condition_margin` helpers.

    Args:
        model: The structural model to check.
        network: An ``EgoSubstrate`` or ``NetworkData`` exposing ``.W`` and ``.X``.
        n_probe: Number of nodes probed for the contraction/locality Jacobian rows.
        locality_tol: Tolerance for the off-neighbourhood derivative.
        uniqueness_tol: Tolerance for multi-start Picard agreement.
        interaction_radius: Hop radius ``r0 >= 1`` the model's ``peer_aggregate`` may
            legitimately reach; the locality check excludes ``B_{r0}(i)`` (default ``1``,
            the 1-hop neighbourhood).
        seed: Seed for the fixed shock draw and the random probe/start.

    Returns:
        A :class:`ModelReport` (truthy iff all checks pass).
    """
    if not isinstance(interaction_radius, int) or interaction_radius < 1:
        raise ValueError(
            f"interaction_radius must be an int >= 1, got {interaction_radius!r}."
        )
    W, X = _extract_w_x(network)
    # Front-load the model's own boundary validation so an invalid (W, X) is rejected with the
    # model's attributable message (e.g. a scalar built-in fed a 2-D covariate raises "supports a
    # scalar covariate only ...") rather than leaking a raw broadcast RuntimeError from the first
    # Picard solve below, which drives best_response directly through the hooks and bypasses
    # forward's guard (D7-REG-checkmodel-2d-x-raw-error).
    model._validate_forward_inputs(W, X)
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
    # In-ball node sets B_{r0}(i) to exclude from the off-ball locality probe. For
    # interaction_radius == 1 this is exactly {i} ∪ 1-hop(i) (the original mask).
    in_ball = _off_ball_masks(indices, neighbors, probe, num_nodes, interaction_radius)

    y_leaf = y_star.clone().requires_grad_(True)
    b_of_y = apply_step(y_leaf)
    max_row_sum = 0.0
    max_off = 0.0
    for i in probe:
        (row,) = torch.autograd.grad(b_of_y[i], y_leaf, retain_graph=True)
        max_row_sum = max(max_row_sum, float(row.abs().sum().item()))
        mask = torch.ones(num_nodes, dtype=torch.bool)
        for j in in_ball[i]:
            mask[j] = False
        if bool(mask.any()):
            max_off = max(max_off, float(row[mask].abs().max().item()))

    # Degenerate probe: with negligible peer interaction at the current params (e.g. a
    # symmetric Interval parameter sitting at its centre ⇒ zero peer coupling), the
    # Jacobian row-sum is ~0, so contraction and locality pass trivially and certify
    # nothing. Surface it loudly on the contraction detail rather than passing silently.
    degenerate = max_row_sum < 1e-8
    contraction_detail = "operator inf-norm max_i sum_j|dB_i/dY_j|"
    if degenerate:
        contraction_detail += (
            "  DEGENERATE: negligible peer interaction at current params; contraction is "
            "uninformative — probe at parameters with active interaction."
        )
    checks.append(CheckResult(
        "contraction_modulus", max_row_sum < 1.0, max_row_sum, 1.0, contraction_detail,
    ))
    if interaction_radius == 1:
        locality_detail = "max |dB_i/dY_j|, j not in 1-hop(i)"
    else:
        locality_detail = f"max |dB_i/dY_j|, j not in B_{interaction_radius}(i) ({interaction_radius}-hop ball)"
    checks.append(CheckResult(
        "locality_A2", max_off < locality_tol, max_off, locality_tol, locality_detail,
    ))

    # --- shock monotonicity dB_i/d eps_i > 0 (diagonal via one VJP) ---
    shocks_leaf = shocks.clone().requires_grad_(True)
    b_eps = model._best_response_step(model.peer_aggregate(W, y_star), X, shocks_leaf)
    (mono,) = torch.autograd.grad(b_eps.sum(), shocks_leaf)
    min_mono = float(mono.min().item())
    checks.append(CheckResult("shock_monotone_U4", min_mono > 0.0, min_mono, 0.0, "min dB_i/d eps_i"))

    # --- uniqueness: a second random start must agree ---
    # The start is the outcome shape (n,), independent of the covariate width d_x. For a
    # 1-D X this is bit-identical to the old torch.randn(X.shape, ...) (same shape, RNG
    # draws, dtype, device); for a (n, d_x) X it no longer crashes the (n,)-shaped solve.
    uniq_thr = max(uniqueness_tol, 100.0 * model.picard_tol)
    with torch.no_grad():
        gen = torch.Generator(device=X.device).manual_seed(seed + 1)
        y_alt, _ = picard(
            apply_step,
            torch.randn(num_nodes, generator=gen, device=X.device, dtype=X.dtype),
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


# =============================================== (M) moment-condition observability
# Condition (M) (Illichmann & Zacchia 2026 finite-moment note, Primitive 2.9):
#     rho_bar^p * lambda < 1   <=>   alpha_p := rho_bar * lambda^(1/p) < 1,
# coupling the preference contraction rho_bar, the graph branching ratio lambda, and the
# shock tail p (with finite p-th moment). It is the *entire stochastic budget* of the
# theory and governs the consistency RATE (m_eff ~ n^gamma; Theorem 11), NOT a per-fit
# admissibility gate: lambda is a property of the graph *ensemble* (Primitive 2.7,
# E|B_d(U)| <= C_B lambda^d) and p of the shock tail, neither of which one fit can
# certify. These helpers SURFACE (M) and the rate for growing-graph experiments; the
# per-fit gate is check_model.
def estimate_branching(
    network,
    *,
    max_depth: int = 3,
    n_roots: int = 256,
    seed: int = 0,
) -> float:
    """Estimate the graph branching ratio ``lambda`` from ball-volume growth.

    Condition (M) bounds ball volumes as ``E|B_d(U)| <= C_B * lambda^d`` (Primitive 2.7),
    so ``lambda`` is the geometric per-hop growth of the closed distance balls. This
    estimates it as the geometric mean of the shell ratios ``|B_{d+1}(u)| / |B_d(u)|``
    over a sample of roots ``u`` and depths ``d >= 1`` (the ``d = 0`` ratio is the
    root-inclusion artifact ``deg(u) + 1`` and is excluded). Ratios from balls that have
    saturated the finite graph (``|B_{d+1}(u)| == |B_d(u)| == n``) are dropped, since a
    saturated ball reports a spurious growth of ``1`` that would deflate ``lambda``.

    Args:
        network: An ``EgoSubstrate`` / ``NetworkData`` exposing ``.W`` (the coalesced
            sparse interaction matrix whose indices are the graph adjacency).
        max_depth: Largest ball radius to materialise (``>= 2`` so a ``d = 1 -> 2`` shell
            ratio exists).
        n_roots: Number of roots sampled for the growth estimate (capped at ``n``).
        seed: Seed for the root sample.

    Returns:
        The estimated branching ratio ``lambda > 0`` (a finite positive float). Falls
        back to the mean degree when no unsaturated shell ratio is available (e.g. a tiny
        or near-complete graph), which is the correct first-moment growth there.

    Raises:
        ValueError: If ``max_depth < 2`` or ``n_roots < 1``.
    """
    if max_depth < 2:
        raise ValueError(f"max_depth must be >= 2 (need a d=1->2 shell), got {max_depth}.")
    if n_roots < 1:
        raise ValueError(f"n_roots must be >= 1, got {n_roots}.")
    W, _ = _extract_w_x(network)
    indices = W.coalesce().indices()
    num_nodes = int(W.shape[0])
    degree = torch.bincount(indices[0], minlength=num_nodes).to(torch.float64)
    mean_degree = float(degree.mean().item())

    adjacency = adjacency_lists_from_edge_index(indices.to(torch.long), num_nodes)
    depth = min(int(max_depth), num_nodes)
    balls = precompute_balls(adjacency, range(depth + 1))

    n_roots = min(int(n_roots), num_nodes)
    gen = torch.Generator().manual_seed(seed)
    roots = torch.randperm(num_nodes, generator=gen)[:n_roots].tolist()

    log_ratios: list[float] = []
    for u in roots:
        for d in range(1, depth):  # d >= 1: skip the root-inclusion artifact at d = 0
            size_d = int(balls[d][u].size)
            size_d1 = int(balls[d + 1][u].size)
            if size_d1 >= num_nodes and size_d >= num_nodes:
                continue  # both saturated: spurious unit growth, uninformative
            if size_d > 0 and size_d1 > size_d:
                log_ratios.append(math.log(size_d1 / size_d))

    if not log_ratios:
        # No unsaturated growing shell (tiny / near-complete graph): the first-moment
        # growth is the mean degree, the GW offspring proxy when the tree saturates fast.
        return max(mean_degree, 1.0)
    return float(math.exp(sum(log_ratios) / len(log_ratios)))


def moment_condition_margin(rho_bar: float, lambda_: float, p: float) -> dict[str, float | bool]:
    """Evaluate the finite-moment binding condition (M) and the consistency rate.

    Condition (M) (Primitive 2.9): ``rho_bar^p * lambda < 1``, equivalently
    ``alpha_p := rho_bar * lambda^(1/p) < 1``. When it holds, the empirical criterion's
    variance decays as ``n^(-gamma)`` (Theorem 11) with effective sample size
    ``m_eff ~ n^gamma`` and

        ``gamma = |log(eta)| / (2 * log(lambda) + |log(eta)|)``,   ``eta := rho_bar * lambda^(1/p)``.

    This SURFACES (M) and the rate; it is not a per-fit gate (lambda/p are ensemble/tail
    properties — use :func:`estimate_branching` for ``lambda``).

    Args:
        rho_bar: Preference/best-response contraction modulus ``rho_bar`` (``> 0``).
        lambda_: Graph branching ratio ``lambda`` (``> 0``; ``estimate_branching`` output).
        p: Shock-tail moment order ``p >= 1`` (finite ``E|eps|^p``).

    Returns:
        ``{value: rho_bar**p * lambda_, holds: value < 1.0, gamma: <rate or nan>}``.
        ``gamma`` is the consistency rate when defined (``eta in (0, 1)`` and
        ``lambda_ > 1``), else ``float('nan')`` (the rate is not defined — e.g. (M) fails,
        or ``lambda_ <= 1`` makes ``2 log lambda + |log eta|`` not the Theorem-11 form).

    Raises:
        ValueError: If ``rho_bar <= 0``, ``lambda_ <= 0``, or ``p < 1``.
    """
    if not (rho_bar > 0.0):
        raise ValueError(f"rho_bar must be strictly positive, got {rho_bar}.")
    if not (lambda_ > 0.0):
        raise ValueError(f"lambda_ must be strictly positive, got {lambda_}.")
    if not (p >= 1.0):
        raise ValueError(f"p must be >= 1, got {p}.")

    value = float(rho_bar) ** float(p) * float(lambda_)
    holds = value < 1.0
    eta = float(rho_bar) * float(lambda_) ** (1.0 / float(p))  # alpha_p = rho_bar * lambda^(1/p)
    if 0.0 < eta < 1.0 and lambda_ > 1.0:
        abs_log_eta = abs(math.log(eta))
        gamma: float = abs_log_eta / (2.0 * math.log(lambda_) + abs_log_eta)
    else:
        gamma = float("nan")
    return {"value": value, "holds": holds, "gamma": gamma}
