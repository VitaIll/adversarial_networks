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

from collections.abc import Callable

import torch
from torch import Tensor

ApplyStep = Callable[[Tensor], Tensor]
ResidualFn = Callable[[Tensor], Tensor]
JacobianFn = Callable[[Tensor], Tensor]


def picard(
    apply_step: ApplyStep,
    y0: Tensor,
    *,
    tol: float,
    max_iter: int,
    fixed_iterations: bool = False,
) -> tuple[Tensor, int]:
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
        ``(Y, iters_used)``.
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")
    Y = y0
    iters_used = max_iter
    for t in range(max_iter):
        Y_next = apply_step(Y)
        if not fixed_iterations:
            max_delta = (Y_next - Y).detach().abs().max().item()
            Y = Y_next
            if max_delta < tol:
                iters_used = t + 1
                break
        else:
            Y = Y_next
    return Y, iters_used


def newton(
    residual_fn: ResidualFn,
    z0: Tensor,
    *,
    tol: float,
    max_iter: int,
    jacobian_fn: JacobianFn | None = None,
    fixed_iterations: bool = False,
) -> tuple[Tensor, int]:
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
        ``(z, iters_used)``.
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")
    z = z0
    iters_used = max_iter
    for s in range(max_iter):
        g = residual_fn(z)
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
        delta = g / g_prime
        z = z - delta
        if not fixed_iterations:
            max_delta = delta.detach().abs().max().item()
            if max_delta < tol:
                iters_used = s + 1
                break
    return z, iters_used


def _implicit_vjp(
    apply_step: ApplyStep,
    y_star: Tensor,
    cotangent: Tensor,
    *,
    tol: float,
    max_iter: int,
) -> Tensor:
    """Solve ``(I - A)^T w = cotangent`` with ``A = dT/dY`` at the fixed point.

    Uses the Neumann recursion ``w_{k+1} = cotangent + A^T w_k`` (which converges
    because ``rho(A) < 1`` under contraction). The matvec ``A^T v`` is one
    reverse-mode pass through a single application of ``apply_step`` at the
    *detached* fixed point ``y_star`` — built once and reused across iterations.
    """
    with torch.enable_grad():
        y_var = y_star.detach().requires_grad_(True)
        t_of_y = apply_step(y_var)

        def at_matvec(v: Tensor) -> Tensor:
            (out,) = torch.autograd.grad(t_of_y, y_var, grad_outputs=v, retain_graph=True)
            return out

        w = cotangent.clone()
        for _ in range(max_iter):
            w_next = cotangent + at_matvec(w)
            if (w_next - w).abs().max().item() < tol:
                w = w_next
                break
            w = w_next
    return w.detach()


def solve_equilibrium(
    apply_step: ApplyStep,
    y0: Tensor,
    *,
    tol: float,
    max_iter: int,
    differentiation: str = "unroll",
    fixed_iterations: bool = False,
) -> tuple[Tensor, int]:
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

    Returns:
        ``(Y, iters_used)``. Under ``"implicit"`` the returned ``Y`` has the
        equilibrium *value* with the implicit structural gradient attached.

    Raises:
        ValueError: If ``differentiation`` is not ``"unroll"`` or ``"implicit"``.
    """
    if differentiation == "unroll":
        return picard(apply_step, y0, tol=tol, max_iter=max_iter, fixed_iterations=fixed_iterations)
    if differentiation != "implicit":
        raise ValueError(
            f"differentiation must be 'unroll' or 'implicit', got {differentiation!r}."
        )

    # Implicit: solve the fixed point off-tape, then attach the adjoint gradient.
    with torch.no_grad():
        y_solved, iters = picard(
            apply_step, y0, tol=tol, max_iter=max_iter, fixed_iterations=fixed_iterations
        )
    y_star = y_solved.detach()

    if not torch.is_grad_enabled():
        return y_star, iters

    t_of_star = apply_step(y_star)  # one on-tape application: carries dT/dθ
    if not t_of_star.requires_grad:
        return y_star, iters  # no trainable parameter reaches the map

    # Forward value is exactly y_star; the gradient flows through t_of_star, whose
    # incoming cotangent is replaced by the adjoint solve (I - A)^{-T} cotangent.
    t_of_star.register_hook(
        lambda grad: _implicit_vjp(apply_step, y_star, grad, tol=tol, max_iter=max_iter)
    )
    y_out = y_star + (t_of_star - t_of_star.detach())
    return y_out, iters
