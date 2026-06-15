"""Differentiable equilibrium solvers — the paper's computational primitive (i).

Three pure functions, ``torch``-only (no ``torch_geometric``):

* :func:`picard` — the geometric fixed-point iteration ``Y_{t+1} = T(Y_t)`` with an
  off-tape convergence check. Under own-concavity + moderate social influence the
  best-response map ``T`` contracts (``rho < 1``), so this converges geometrically
  to the unique equilibrium (Section 2; Banach).
* :func:`newton` — a *vectorised scalar* Newton solve for a per-node first-order
  condition ``g_i(z_i) = 0``. Because each node's FOC is scalar in its own action
  (the peer aggregate is fixed within a Picard step), the Jacobian is **diagonal**;
  it is supplied analytically (``jacobian_fn``, fast/bit-stable) or obtained by one
  reverse-mode pass (``autograd.grad(g.sum(), z)`` returns exactly the diagonal).
* :func:`solve_equilibrium` — wraps :func:`picard` with the structural-gradient
  strategy. ``"unroll"`` differentiates through the executed iteration (paper-
  faithful, what the built-ins are tested under). ``"implicit"`` solves the fixed
  point off-tape and recovers the structural gradient by the implicit-function
  theorem ``dY/dθ = (I − A)^{-1} dT/dθ`` (eq. 2.1; the deep-equilibrium adjoint):
  ``O(n)`` memory, no unrolled graph, exact at the fixed point.

The two strategies agree to gradient tolerance at a converged fixed point.

References:
    Illichmann & Zacchia (2026), Algorithm 1 and eq. (2.1). Bai, Kolter & Koltun
    (2019), *Deep Equilibrium Models* (the implicit-differentiation adjoint).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

ApplyStep = Callable[[Tensor], Tensor]
ResidualFn = Callable[[Tensor], Tensor]
JacobianFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class SolveResult:
    """Outcome of an equilibrium / FOC solve, with convergence made observable.

    Algorithm 1 terminates on ``||y(t+1) - y(t)|| < tau OR t = T_max``: the cap is a
    *normal* computable-approximation state, not a hard failure, and the **residual**
    is the quantity that distinguishes a genuine stop from a truncation. Reporting
    ``iters`` alone is ambiguous (a real stop at the last allowed iteration is
    indistinguishable from a cap hit), so this carries both the residual and the
    explicit ``converged`` flag (whether the tolerance test actually fired).

    Attributes:
        Y: The solved iterate ``(n,)`` (with the structural gradient attached when
            grad is enabled).
        iters: Iterations actually executed.
        residual: Final ``max|Y_{t+1} - Y_t|`` (picard) / ``max|delta|`` (newton).
            ``0.0`` for a ``fixed_iterations`` run (no convergence test was run).
        converged: Whether the tolerance test fired (``False`` on a cap hit, and on
            a ``fixed_iterations`` run where no test is performed).
    """

    Y: Tensor
    iters: int
    residual: float
    converged: bool

    def __iter__(self):
        # Backwards-compatible 2-tuple unpacking ``Y, iters = solve(...)``.
        yield self.Y
        yield self.iters


@dataclass(frozen=True)
class _AdjointResult:
    """Outcome of the implicit-differentiation cotangent (adjoint) Neumann solve.

    The adjoint solves ``(I - A)^T w = cotangent`` by a truncated Neumann recursion, so —
    exactly like the forward Picard — a too-small cap returns an under-solved ``w`` whose
    error is ``O(rho^K)``. Reporting only ``w`` would make that under-solve silent (it fires
    during ``backward``), so this carries the final **relative** residual and whether the
    relative-tol test actually fired, mirroring :class:`SolveResult` for the forward solve.

    Attributes:
        w: The solved cotangent ``(I - A)^{-T} cotangent`` (detached).
        residual: Final relative residual ``max|w_{k+1}-w_k| / (max|w_k| + 1e-12)`` at the
            last executed Neumann step (scale-invariant in the cotangent magnitude).
        converged: Whether the relative-tol test fired (``False`` on a cap hit).
    """

    w: Tensor
    residual: float
    converged: bool


class EquilibriumNotConverged(RuntimeError):
    """Raised (opt-in) when an equilibrium solve hits its cap without converging.

    Carries the attributable quantities (the final ``residual``, the ``tol`` it was
    tested against, and the ``max_iter`` cap) so the failure is self-explaining.
    """

    def __init__(self, residual: float, tol: float, max_iter: int) -> None:
        self.residual = float(residual)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        super().__init__(
            f"equilibrium did not converge: residual {self.residual:.4g} >= tol "
            f"{self.tol:.4g} after the full max_iter={self.max_iter} budget; the "
            "returned iterate is a truncated (non-equilibrium) approximation. Raise "
            "max_iter (paper T = O(log(1/tol)/|log rho|); high-contraction rho near 1 "
            "needs a larger cap) or verify contraction (rho < 1) via check_model."
        )


def _check_map_output(value: object, y0: Tensor, *, hook: str) -> None:
    """First-iteration boundary guard: the map's output must match the iterate shape.

    A grad-connected scalar (or otherwise mis-shaped) ``apply_step``/``residual_fn``
    would silently broadcast through ``Y_next - Y`` (picard) or ``g / g'`` (newton),
    defeating the ``||y(t+1) - y(t)||`` stopping test (which is only meaningful for
    matching ``(n,)`` shapes). Reject it at the boundary, naming the hook, rather than
    letting it collapse/broadcast silently. Cheap — evaluated on the first call only.
    """
    if not isinstance(value, Tensor):
        raise ValueError(f"{hook} must return a torch.Tensor, got {type(value).__name__}.")
    if value.shape != y0.shape:
        raise ValueError(
            f"{hook} must return a tensor matching the iterate shape "
            f"{tuple(y0.shape)}, got {tuple(value.shape)} (a scalar/wrong-shape map "
            "would broadcast silently and defeat the ||y(t+1)-y(t)|| convergence test)."
        )


def picard(
    apply_step: ApplyStep,
    y0: Tensor,
    *,
    tol: float,
    max_iter: int,
    fixed_iterations: bool = False,
) -> SolveResult:
    """Solve the fixed point ``Y = apply_step(Y)`` by Picard iteration.

    Records on the autograd tape iff the caller has grad enabled (the convergence
    check is always off-tape, via ``.detach()``).

    Args:
        apply_step: The best-response map ``T`` (one Picard update ``Y -> T(Y)``).
        y0: Initial iterate ``(n,)``.
        tol: Stop when ``max|Y_{t+1} - Y_t| < tol`` (ignored if ``fixed_iterations``).
        max_iter: Maximum / exact iteration count.
        fixed_iterations: If true, always run exactly ``max_iter`` steps (no early
            stop) — used to make autograd graphs shape-stable for ``gradcheck``.

    Returns:
        A :class:`SolveResult` (``Y``, ``iters``, ``residual``, ``converged``).
        Iterable as the legacy ``(Y, iters)`` 2-tuple.
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")
    Y = y0
    iters_used = max_iter
    residual = 0.0
    converged = False
    for t in range(max_iter):
        Y_next = apply_step(Y)
        if t == 0:
            _check_map_output(Y_next, y0, hook="apply_step")
        if not fixed_iterations:
            max_delta = (Y_next - Y).detach().abs().max().item()
            Y = Y_next
            residual = max_delta
            if max_delta < tol:
                iters_used = t + 1
                converged = True
                break
        else:
            Y = Y_next
    return SolveResult(Y=Y, iters=iters_used, residual=float(residual), converged=converged)


def newton(
    residual_fn: ResidualFn,
    z0: Tensor,
    *,
    tol: float,
    max_iter: int,
    jacobian_fn: JacobianFn | None = None,
    fixed_iterations: bool = False,
) -> SolveResult:
    """Vectorised scalar Newton solve of the per-node FOC ``residual_fn(z) = 0``.

    The residual must be *elementwise* in ``z`` (node ``i``'s residual depends only
    on ``z_i`` given a fixed peer aggregate), so its Jacobian is diagonal:

    * ``jacobian_fn is not None`` → use the supplied analytic diagonal ``g'(z)``
      (the built-in effort game's route: fast and bit-stable).
    * ``jacobian_fn is None`` → AD-diagonal: ``autograd.grad(g.sum(), z)`` returns
      ``[dg_i/dz_i]`` in one reverse pass (the ``foc_residual`` hook's route — the
      user writes only the residual).

    Args:
        residual_fn: The per-node FOC residual ``g(z)`` (elementwise in ``z``).
        z0: Initial iterate ``(n,)``.
        tol: Stop when ``max|delta| < tol`` (ignored if ``fixed_iterations``).
        max_iter: Maximum / exact iteration count.
        jacobian_fn: Optional analytic diagonal Jacobian ``g'(z)``.
        fixed_iterations: If true, run exactly ``max_iter`` steps (gradcheck-stable).

    Returns:
        A :class:`SolveResult` (``z`` in ``Y``, ``iters``, ``residual``,
        ``converged``). Iterable as the legacy ``(z, iters)`` 2-tuple.
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")
    z = z0
    iters_used = max_iter
    residual = 0.0
    converged = False
    for s in range(max_iter):
        g = residual_fn(z)
        if s == 0:
            _check_map_output(g, z0, hook="residual_fn")
        if jacobian_fn is not None:
            g_prime = jacobian_fn(z)
        elif z.requires_grad:
            # On-tape AD-diagonal: keep g' differentiable wrt params so the
            # unrolled Newton carries the structural gradient (one reverse pass
            # over the elementwise residual returns exactly [dg_i/dz_i]).
            (g_prime,) = torch.autograd.grad(
                g.sum(), z, create_graph=True, retain_graph=True
            )
        else:
            # Off-tape (e.g. the implicit forward solve, or a no-grad context):
            # a detached diagonal value via a local leaf.
            with torch.enable_grad():
                z_leaf = z.detach().requires_grad_(True)
                (gp,) = torch.autograd.grad(residual_fn(z_leaf).sum(), z_leaf)
            g_prime = gp.detach()
        # Reject a degenerate diagonal Jacobian at the boundary (before dividing): a
        # non-finite or zero g'(z) makes the per-node Newton step g/g' undefined. This
        # signals a degenerate FOC — own-concavity (U2) or the monotone shock channel
        # (U4) is violated — rather than letting the step blow up silently downstream.
        if not torch.isfinite(g_prime).all() or bool((g_prime == 0).any()):
            raise ValueError(
                "Newton: the per-node FOC residual has a non-finite or zero derivative "
                "g'(z), so the Newton step g/g' is undefined; the FOC is degenerate "
                "(own-concavity (U2) / monotone shock channel (U4) likely violated). "
                "Run check_model to localise the failing condition."
            )
        delta = g / g_prime
        z = z - delta
        if not fixed_iterations:
            max_delta = delta.detach().abs().max().item()
            residual = max_delta
            if max_delta < tol:
                iters_used = s + 1
                converged = True
                break
    return SolveResult(Y=z, iters=iters_used, residual=float(residual), converged=converged)


def _implicit_vjp(
    apply_step: ApplyStep,
    y_star: Tensor,
    cotangent: Tensor,
    *,
    tol: float,
    max_iter: int,
) -> _AdjointResult:
    """Solve ``(I - A)^T w = cotangent`` with ``A = dT/dY`` at the fixed point.

    Uses the Neumann recursion ``w_{k+1} = cotangent + A^T w_k`` (which converges
    because ``rho(A) < 1`` under contraction). The matvec ``A^T v`` is one
    reverse-mode pass through a single application of ``apply_step`` at the
    *detached* fixed point ``y_star`` — built once and reused across iterations.

    The adjoint converges on a **relative** residual (scale-invariant in the
    cotangent: ``max|w_{k+1} - w_k| < tol * (max|w_k| + 1e-12)``), decoupled from the
    outcome-scaled Picard tolerance — the incoming gradient's arbitrary magnitude must
    not set an absolute stopping threshold for the cotangent solve. ``max_iter`` is the
    adjoint's *own* iteration budget (see :func:`solve_equilibrium`'s
    ``adjoint_max_iter``): unrolling ``T`` Picard steps truncates the Neumann
    expansion of ``(I - A)^{-1}`` at order ``T``, so the cotangent solve needs at least
    as many terms as the forward solve — a small inherited cap would under-solve it
    near ``rho -> 1``.

    Returns:
        An :class:`_AdjointResult` carrying the solved ``w`` (detached), the final
        **relative** residual, and whether the relative-tol test fired — so a cap-hit
        under-solve (which happens during ``backward``) is observable by the caller, not
        silent.
    """
    with torch.enable_grad():
        y_var = y_star.detach().requires_grad_(True)
        t_of_y = apply_step(y_var)

        def at_matvec(v: Tensor) -> Tensor:
            (out,) = torch.autograd.grad(t_of_y, y_var, grad_outputs=v, retain_graph=True)
            return out

        w = cotangent.clone()
        residual = 0.0
        converged = False
        for _ in range(max_iter):
            w_next = cotangent + at_matvec(w)
            # Relative step residual, scale-invariant in the cotangent magnitude (the same
            # quantity the tolerance test compares), so it is a meaningful convergence
            # measure regardless of the incoming gradient's arbitrary scale.
            residual = (w_next - w).abs().max().item() / (w.abs().max().item() + 1e-12)
            w = w_next
            if residual < tol:
                converged = True
                break
    return _AdjointResult(w=w.detach(), residual=float(residual), converged=converged)


def solve_equilibrium(
    apply_step: ApplyStep,
    y0: Tensor,
    *,
    tol: float,
    max_iter: int,
    differentiation: str = "unroll",
    fixed_iterations: bool = False,
    adjoint_max_iter: int | None = None,
) -> SolveResult:
    """Solve the equilibrium and select the structural-gradient strategy.

    Args:
        apply_step: The best-response map ``T`` (one Picard update). Must close over
            a *single* shock draw so the map is deterministic across iterations.
        y0: Initial iterate ``(n,)``.
        tol: Picard convergence tolerance.
        max_iter: Picard iteration cap.
        differentiation: ``"unroll"`` (autograd through the executed Picard) or
            ``"implicit"`` (implicit-function-theorem adjoint at the fixed point).
        fixed_iterations: Forwarded to :func:`picard` (gradcheck-stable graphs).
        adjoint_max_iter: Iteration cap for the ``"implicit"`` adjoint (cotangent)
            Neumann solve. Defaults to ``max(max_iter, 200)`` so the adjoint is not
            silently truncated by the (possibly small) forward Picard budget — the
            cotangent solve must reach the same fidelity as the forward equilibrium.
            Unused under ``"unroll"``.

    Returns:
        A :class:`SolveResult` carrying the forward-solve ``residual`` / ``converged``
        (the implicit path propagates the forward Picard's values). Under
        ``"implicit"`` the returned ``Y`` has the equilibrium *value* with the
        implicit structural gradient attached. Iterable as ``(Y, iters)``.

    Raises:
        ValueError: If ``differentiation`` is not ``"unroll"`` or ``"implicit"``.
    """
    if differentiation == "unroll":
        return picard(apply_step, y0, tol=tol, max_iter=max_iter, fixed_iterations=fixed_iterations)
    if differentiation != "implicit":
        raise ValueError(
            f"differentiation must be 'unroll' or 'implicit', got {differentiation!r}."
        )

    adjoint_iter = int(adjoint_max_iter) if adjoint_max_iter is not None else max(max_iter, 200)

    # Implicit: solve the fixed point off-tape, then attach the adjoint gradient. The
    # forward solve's residual/converged are propagated so the caller sees the same
    # convergence observability as the unroll path.
    with torch.no_grad():
        forward = picard(
            apply_step, y0, tol=tol, max_iter=max_iter, fixed_iterations=fixed_iterations
        )
    y_star = forward.Y.detach()

    if not torch.is_grad_enabled():
        return SolveResult(
            Y=y_star, iters=forward.iters, residual=forward.residual, converged=forward.converged
        )

    t_of_star = apply_step(y_star)  # one on-tape application: carries dT/dθ
    if not t_of_star.requires_grad:
        return SolveResult(  # no trainable parameter reaches the map
            Y=y_star, iters=forward.iters, residual=forward.residual, converged=forward.converged
        )

    # Forward value is exactly y_star; the gradient flows through t_of_star, whose
    # incoming cotangent is replaced by the adjoint solve (I - A)^{-T} cotangent. The
    # adjoint uses its own cap (adjoint_iter), not the forward Picard budget.
    #
    # The hook fires during backward, so an under-solved cotangent (cap hit) would
    # otherwise be silent: surface it ONCE per solve as an attributable RuntimeWarning
    # naming the final relative residual and the cap, since the implicit structural
    # gradient is then inaccurate (the same O(rho^K) Neumann truncation the forward solve
    # incurs). The flag dedupes across the (possibly many) cotangent rows of one backward.
    adjoint_warned = [False]

    def adjoint_hook(grad: Tensor) -> Tensor:
        result = _implicit_vjp(apply_step, y_star, grad, tol=tol, max_iter=adjoint_iter)
        if not result.converged and not adjoint_warned[0]:
            adjoint_warned[0] = True
            warnings.warn(
                "implicit-differentiation adjoint (cotangent Neumann solve) did not "
                f"converge: final relative residual {result.residual:.4g} >= tol {tol:.4g} "
                f"after the full adjoint_max_iter={adjoint_iter} budget; the implicit "
                "structural gradient is an O(rho^K) truncation and may be inaccurate. Raise "
                "adjoint_max_iter (the cotangent solve needs at least as many Neumann terms "
                "as the forward equilibrium; high-contraction rho near 1 needs a larger cap) "
                "or verify contraction (rho < 1) via check_model.",
                RuntimeWarning,
                stacklevel=2,
            )
        return result.w

    t_of_star.register_hook(adjoint_hook)
    y_out = y_star + (t_of_star - t_of_star.detach())
    return SolveResult(
        Y=y_out, iters=forward.iters, residual=forward.residual, converged=forward.converged
    )
