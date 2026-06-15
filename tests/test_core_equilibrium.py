"""Tests for the differentiable equilibrium core (picard / newton / solve_equilibrium).

Pins the new numeric kernels in isolation (float64): Picard converges to the dense
linear solve, the analytic and AD-diagonal Newton agree, and the ``"unroll"`` and
``"implicit"`` structural-gradient strategies agree (value and gradient) at a
converged fixed point, with ``gradcheck`` on the unrolled solve.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch
from torch_geometric.utils import from_networkx, to_undirected

from adversarial_networks.core.equilibrium import (
    SolveResult,
    _implicit_vjp,
    newton,
    picard,
    solve_equilibrium,
)
from adversarial_networks.core.graph import row_stochastic_weights


def _w64(n: int = 12, seed: int = 0) -> torch.Tensor:
    graph = nx.barabasi_albert_graph(n, 2, seed=seed)
    ei = to_undirected(from_networkx(graph).edge_index, num_nodes=n).contiguous()
    return row_stochastic_weights(ei, n).to(torch.float64)


def test_picard_matches_dense_linear_solve() -> None:
    torch.manual_seed(0)
    n = 12
    W = _w64(n)
    X = torch.randn(n, dtype=torch.float64)
    shocks = torch.randn(n, dtype=torch.float64)
    beta, gamma = torch.tensor(0.5, dtype=torch.float64), torch.tensor(1.3, dtype=torch.float64)
    base = gamma * X + shocks

    def step(Y):
        return beta * torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1) + base

    Y, iters = picard(step, torch.zeros(n, dtype=torch.float64), tol=1e-13, max_iter=500)
    dense = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - beta * W.to_dense(), base)
    assert (Y - dense).abs().max().item() < 1e-9
    assert 1 <= iters <= 500


def test_newton_analytic_equals_ad_diagonal() -> None:
    n = 10
    torch.manual_seed(1)
    X = torch.randn(n, dtype=torch.float64)
    b = 1.3 * X + torch.randn(n, dtype=torch.float64)
    lam, mu, r = (torch.tensor(v, dtype=torch.float64) for v in (0.8, 0.4, 1.0))

    def residual(z):
        return (1 + lam) * z - mu * r * torch.exp(-r * z) - b

    def jacobian(z):
        return (1 + lam) + mu * r * r * torch.exp(-r * z)

    z0 = b / (1 + lam)
    z_analytic, _ = newton(residual, z0, tol=1e-13, max_iter=40, jacobian_fn=jacobian)
    z_ad, _ = newton(residual, z0, tol=1e-13, max_iter=40, jacobian_fn=None)
    assert (z_analytic - z_ad).abs().max().item() < 1e-10
    assert residual(z_analytic).abs().max().item() < 1e-10


def test_unroll_and_implicit_agree_on_value_and_gradient() -> None:
    n = 10
    W = _w64(n, seed=2)
    torch.manual_seed(2)
    X = torch.randn(n, dtype=torch.float64)
    shocks = torch.randn(n, dtype=torch.float64)
    weights = torch.linspace(0.5, 1.5, n, dtype=torch.float64)

    lam = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    mu = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def apply_step(Y):
        b = lam * torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1) + gamma * X + shocks
        z = b / (1 + lam)
        for _ in range(14):
            e = torch.exp(-z)
            z = z - ((1 + lam) * z - mu * e - b) / ((1 + lam) + mu * e)
        return z

    def run(mode):
        for p in (lam, gamma, mu):
            p.grad = None
        Y, _ = solve_equilibrium(apply_step, torch.zeros(n, dtype=torch.float64),
                                 tol=1e-13, max_iter=300, differentiation=mode)
        (Y * weights).sum().backward()
        return Y.detach(), [float(p.grad) for p in (lam, gamma, mu)]

    y_unroll, g_unroll = run("unroll")
    y_implicit, g_implicit = run("implicit")
    assert (y_unroll - y_implicit).abs().max().item() < 1e-9
    assert max(abs(a - b) for a, b in zip(g_unroll, g_implicit)) < 1e-7


def test_gradcheck_unrolled_solve() -> None:
    n = 8
    W = _w64(n, seed=3)
    torch.manual_seed(3)
    X = torch.randn(n, dtype=torch.float64)
    shocks = torch.randn(n, dtype=torch.float64)

    def f(beta, gamma):
        base = gamma * X + shocks

        def step(Y):
            return beta * torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1) + base

        Y, _ = solve_equilibrium(step, torch.zeros(n, dtype=torch.float64),
                                 tol=0.0, max_iter=80, differentiation="unroll", fixed_iterations=True)
        return Y

    beta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (beta, gamma), eps=1e-6, atol=1e-5, rtol=1e-4)


# ------------------------------------------- Newton degenerate-Jacobian boundary guard
def test_newton_rejects_zero_jacobian_analytic() -> None:
    """An analytic ``jacobian_fn`` that is identically zero makes the Newton step g/g'
    undefined; the boundary guard must raise an attributable ValueError before dividing."""
    z0 = torch.ones(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite or zero"):
        newton(
            lambda z: z**2 - 2.0,
            z0,
            tol=1e-10,
            max_iter=5,
            jacobian_fn=lambda z: torch.zeros_like(z),
        )


def test_newton_rejects_zero_jacobian_ad() -> None:
    """A residual whose derivative is identically zero (AD g'=0) — but still graph-
    connected — must trip the same guard on the AD-diagonal route."""
    z0 = torch.ones(5, dtype=torch.float64)  # no requires_grad -> off-tape AD branch
    with pytest.raises(ValueError, match="own-concavity"):
        newton(lambda z: 0.0 * z + 1.0, z0, tol=1e-10, max_iter=5)


def test_newton_rejects_non_finite_jacobian() -> None:
    """A non-finite diagonal derivative is equally degenerate and must be rejected."""
    z0 = torch.ones(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="check_model"):
        newton(
            lambda z: z**2 - 2.0,
            z0,
            tol=1e-10,
            max_iter=5,
            jacobian_fn=lambda z: torch.full_like(z, float("nan")),
        )


# --------------------------------------------- implicit-adjoint scale-invariant tolerance
def test_implicit_vjp_relative_tol_is_scale_invariant() -> None:
    """The cotangent Neumann solve must converge on a *relative* residual: a tiny
    cotangent (where the old absolute tol stopped after one step at ~9% error) is solved
    to the true ``(I - A)^{-T} c``, and the normalised solution is invariant to the
    cotangent magnitude across many orders of magnitude."""
    torch.manual_seed(0)
    n = 6
    A = torch.randn(n, n, dtype=torch.float64) * 0.12  # contraction: ||A|| well below 1
    const = torch.randn(n, dtype=torch.float64)
    eye = torch.eye(n, dtype=torch.float64)
    y_star = torch.linalg.solve(eye - A, const)  # fixed point of T(y) = A y + const

    def apply_step(y: torch.Tensor) -> torch.Tensor:
        return A @ y + const

    torch.manual_seed(1)
    base_c = torch.randn(n, dtype=torch.float64)
    tol, max_iter = 1e-8, 200

    normalised: list[torch.Tensor] = []
    for scale in (1e-10, 1.0, 1e8):
        c = scale * base_c
        result = _implicit_vjp(apply_step, y_star, c, tol=tol, max_iter=max_iter)
        assert result.converged is True  # well-conditioned (||A|| << 1): tol test fires
        assert result.residual < tol
        w = result.w
        truth = torch.linalg.solve((eye - A).T, c)  # exact (I - A)^{-T} c
        rel_err = (w - truth).abs().max().item() / truth.abs().max().item()
        assert rel_err < 1e-6, f"scale={scale}: rel_err {rel_err} (absolute tol would stop early)"
        normalised.append(w / scale)

    # scale-invariance: the normalised solution is identical regardless of cotangent size.
    for w_norm in normalised[1:]:
        assert (w_norm - normalised[0]).abs().max().item() < 1e-8


# --------------------------------------- convergence observability (residual/converged)
def test_picard_reports_residual_and_converged_flag() -> None:
    """picard returns a SolveResult carrying the final residual and whether the tol test
    fired — the residual is the quantity that distinguishes a genuine stop from a cap
    truncation (iters alone is ambiguous)."""
    # T(Y) = 0.5 Y + 1, fixed point Y* = 2. From Y0=0 the residual is 0.5^t.
    def step(Y):
        return 0.5 * Y + 1.0

    y0 = torch.zeros(3, dtype=torch.float64)
    result = picard(step, y0, tol=1e-9, max_iter=200)
    assert isinstance(result, SolveResult)
    assert result.converged is True
    assert result.residual < 1e-9
    assert (result.Y - 2.0).abs().max().item() < 1e-8
    # legacy 2-tuple unpacking still works
    Y, iters = result
    assert Y is result.Y and iters == result.iters


def test_picard_converges_exactly_at_last_iteration_is_not_a_cap_hit() -> None:
    """A genuine convergence whose tol test fires on the LAST allowed iteration must
    report converged=True (NOT a cap hit) even though iters == max_iter — the old
    iters>=cap heuristic false-positived here."""
    # T(Y)=0.5Y+1: residual 0.5^t. At max_iter=20, tol=2.861e-6 the test fires at t=19
    # (0.5^19=1.907e-6 < tol; 0.5^18=3.81e-6 > tol), i.e. exactly on iteration 20.
    def step(Y):
        return 0.5 * Y + 1.0

    result = picard(step, torch.zeros(2, dtype=torch.float64), tol=2.861e-6, max_iter=20)
    assert result.iters == 20  # the last allowed iteration
    assert result.converged is True  # but it genuinely converged, not a cap truncation
    assert result.residual < 2.861e-6


def test_picard_rho095_at_small_cap_flags_nonconvergence_with_residual() -> None:
    """A rho=0.95 contraction at a cap too small to converge is flagged non-converged
    with the residual surfaced (residual >= tol), not silently truncated."""
    rho = 0.95
    def step(Y):
        return rho * Y + 1.0

    result = picard(step, torch.zeros(4, dtype=torch.float64), tol=1e-6, max_iter=50)
    assert result.converged is False
    assert result.iters == 50
    assert result.residual >= 1e-6  # the distinguishing quantity is surfaced and large


def test_newton_reports_residual_and_converged_flag() -> None:
    """newton likewise returns residual (max|delta|) and converged."""
    n = 6
    torch.manual_seed(0)
    b = torch.randn(n, dtype=torch.float64)

    def residual(z):
        return z - b  # trivial: one Newton step lands exactly, delta -> 0

    res = newton(residual, torch.zeros(n, dtype=torch.float64), tol=1e-9, max_iter=20)
    assert isinstance(res, SolveResult)
    assert res.converged is True
    assert res.residual < 1e-9
    assert (res.Y - b).abs().max().item() < 1e-9


def test_newton_nonconvergence_flagged_with_residual() -> None:
    """A Newton solve capped before convergence reports converged=False + residual."""
    # g(z) = z^3 - 2 has a slowly-resolving Newton from a poor start under a tiny cap.
    def residual(z):
        return z**3 - 2.0

    def jacobian(z):
        return 3.0 * z**2

    res = newton(
        residual,
        torch.full((3,), 50.0, dtype=torch.float64),
        tol=1e-12,
        max_iter=2,
        jacobian_fn=jacobian,
    )
    assert res.converged is False
    assert res.iters == 2
    assert res.residual >= 1e-12


# ------------------------------------------------ first-iteration output-shape boundary
def test_picard_rejects_scalar_collapsing_map() -> None:
    """A map that collapses (n,) -> scalar would broadcast silently through Y_next-Y and
    defeat the convergence test; the first-iteration boundary guard rejects it."""
    with pytest.raises(ValueError, match="apply_step"):
        picard(lambda Y: Y.mean(), torch.zeros(4, dtype=torch.float64), tol=1e-9, max_iter=5)
    with pytest.raises(ValueError, match=r"matching the iterate shape"):
        picard(
            lambda Y: torch.tensor(0.5, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
            tol=1e-9,
            max_iter=5,
        )


def test_newton_rejects_scalar_collapsing_residual() -> None:
    """A residual_fn that returns a scalar (or wrong shape) would broadcast through g/g';
    the first-iteration boundary guard rejects it before the Newton step."""
    with pytest.raises(ValueError, match="residual_fn"):
        newton(
            lambda z: z.sum(),  # (n,) -> scalar
            torch.zeros(4, dtype=torch.float64),
            tol=1e-9,
            max_iter=5,
            jacobian_fn=lambda z: torch.ones_like(z).sum(),
        )


# -------------------------------------------- implicit adjoint has its own iteration cap
def _adjoint_grad(rho: float, *, max_iter: int, adjoint_max_iter: int | None, tol: float = 1e-9):
    """Run the implicit solve at contraction ``rho`` with the given caps and return
    ``(d(w·Y*)/d scale, truth)`` — the structural gradient flows through the adjoint.

    ``tol`` is BOTH the forward Picard and the adjoint Neumann relative tolerance. The
    default ``1e-9`` is a relative residual that float64 can actually reach for a
    well-conditioned (moderate-rho) solve, so a converged solve reports ``converged=True``
    (a tol below the float-precision step-residual floor, e.g. ``1e-13``, would never fire
    the test even when the gradient is accurate)."""
    n = 5
    A = rho * torch.eye(n, dtype=torch.float64)
    const = torch.ones(n, dtype=torch.float64)
    eye = torch.eye(n, dtype=torch.float64)
    y_star = torch.linalg.solve(eye - A, const)
    weights = torch.linspace(0.5, 1.5, n, dtype=torch.float64)
    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

    def apply_step(Y):
        return A @ Y + const * scale

    result = solve_equilibrium(
        apply_step, y_star.clone(), tol=tol, max_iter=max_iter,
        differentiation="implicit", adjoint_max_iter=adjoint_max_iter,
    )
    (result.Y * weights).sum().backward()
    truth = float(weights @ torch.linalg.solve(eye - A, const))
    return float(scale.grad), truth


def test_implicit_adjoint_uses_own_cap_not_small_forward_cap() -> None:
    """The implicit adjoint Neumann solve must use its OWN cap (default
    ``max(max_iter, 200)``), not the (small) forward Picard ``max_iter``: at rho=0.9 a
    forward cap of 5 truncates the cotangent badly, but the default adjoint cap solves it
    to the true ``(I - A)^{-T}`` cotangent."""
    # Default adjoint cap (max(5, 200) = 200) at rho=0.9 converges to the 1e-9 relative tol.
    grad_default, truth = _adjoint_grad(0.9, max_iter=5, adjoint_max_iter=None)
    assert abs(grad_default - truth) < 1e-6, "adjoint under-solved with the default cap"

    # If the adjoint had instead inherited the tiny forward cap (5), it would truncate the
    # Neumann series badly -> a large error. This pins that the cap is the lever. (The
    # deliberate under-solve emits the D2-04 non-convergence warning, which is expected
    # here and suppressed so it does not pollute the run.)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grad_truncated, _ = _adjoint_grad(0.9, max_iter=5, adjoint_max_iter=5)
    assert abs(grad_truncated - truth) > 1e-3, (
        "a 5-term adjoint should be badly truncated; the default cap must not equal it"
    )


# ------------------------------------------ adjoint non-convergence observability (D2-04)
def test_adjoint_nonconvergence_warns_with_residual_and_cap() -> None:
    """A near-1 contraction (rho=0.99) under a tiny adjoint cap leaves the cotangent
    Neumann solve under-solved; because the hook fires during backward, the engine must
    surface ONE attributable RuntimeWarning naming the final relative residual and the
    adjoint_max_iter cap (the implicit structural gradient is otherwise silently wrong)."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # rho=0.99 needs ~hundreds of Neumann terms; adjoint_max_iter=5 cannot converge.
        grad, truth = _adjoint_grad(0.99, max_iter=300, adjoint_max_iter=5)
    adjoint_warnings = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "adjoint" in str(w.message)
        and "did not converge" in str(w.message)
    ]
    assert len(adjoint_warnings) == 1  # surfaced exactly once across the backward pass
    msg = str(adjoint_warnings[0].message)
    assert "adjoint_max_iter=5" in msg  # the cap is named
    assert "relative residual" in msg  # the residual is named
    # The under-solve is real: the truncated adjoint gradient is far from the truth.
    assert abs(grad - truth) > 1e-3


def test_adjoint_well_conditioned_solve_does_not_warn() -> None:
    """A well-conditioned solve (moderate rho, ample adjoint cap) converges, so NO adjoint
    non-convergence warning is emitted (the warning is attributable, not noisy)."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grad, truth = _adjoint_grad(0.9, max_iter=5, adjoint_max_iter=None)  # default cap 200
    assert not [
        w for w in caught if "adjoint" in str(w.message) and "did not converge" in str(w.message)
    ]
    assert abs(grad - truth) < 1e-6  # and the gradient is accurate
