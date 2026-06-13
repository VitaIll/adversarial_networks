"""Structural effort-game generator with differentiable Picard + Newton equilibrium solve.

This module implements the nonlinear effort-game generator used by the adversarial
structural estimator. Equilibrium outcomes are solved numerically using Picard
fixed-point iteration with a vectorized Newton inner loop, and gradients are
computed by standard PyTorch reverse-mode AD through the executed iterations.

References:
    - Kaji, Manresa & Pouliot (2023). "An Adversarial Approach to Structural
      Estimation." Econometrica, 91(6), 2041-2063.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EffortGameGenerator(nn.Module):
    """Structural simulator for nonlinear network effort-game equilibrium.

    The generator solves node outcomes Y from the first-order condition
    composition described in the effort-game design document, using:
        1) Picard iteration over neighborhood interactions, and
        2) vectorized Newton updates for each Picard step.

    All solver arithmetic remains on the autograd tape. In adaptive mode
    (`fixed_iterations=False`), only convergence checks are off-tape via
    `.detach().abs().max().item()`.
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
    ) -> None:
        super().__init__()

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
        if not (picard_tol > 0.0):
            raise ValueError("picard_tol must be strictly positive.")
        if picard_max <= 0:
            raise ValueError("picard_max must be positive.")
        if not (newton_tol > 0.0):
            raise ValueError("newton_tol must be strictly positive.")
        if newton_max <= 0:
            raise ValueError("newton_max must be positive.")

        self.lambda_max = float(lambda_max)
        self.picard_tol = float(picard_tol)
        self.picard_max = int(picard_max)
        self.newton_tol = float(newton_tol)
        self.newton_max = int(newton_max)
        self.fixed_iterations = bool(fixed_iterations)

        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

        scaled = torch.tensor(init_lambda / self.lambda_max, dtype=torch.float32)
        scaled = scaled.clamp(1e-6, 1.0 - 1e-6)
        self.raw_lambda = nn.Parameter(torch.logit(scaled))

        self.log_mu = nn.Parameter(
            torch.log(torch.tensor(float(init_mu), dtype=torch.float32))
        )

        if fix_r is not None:
            self._fixed_r: float | None = float(fix_r)
        else:
            self._fixed_r = None
            self.log_r = nn.Parameter(
                torch.log(torch.tensor(float(init_r), dtype=torch.float32))
            )

        if fix_sigma_sq is not None:
            self._fixed_sigma_sq: float | None = float(fix_sigma_sq)
        else:
            self._fixed_sigma_sq = None
            self.log_sigma_sq = nn.Parameter(
                torch.tensor(float(init_log_sigma_sq), dtype=torch.float32)
            )

        self.last_picard_iterations: int = 0
        self.last_newton_max_iters: int = 0

    def _validate_forward_inputs(self, W: Tensor, X: Tensor) -> None:
        """Validate input tensors for forward pass.

        Args:
            W: Network weight matrix.
            X: Node covariate vector.

        Raises:
            TypeError: If W is not sparse COO or X is not floating-point.
            ValueError: If shapes, dtypes, or devices are incompatible.
        """
        if not isinstance(W, Tensor):
            raise TypeError("W must be a torch.Tensor.")
        if not W.is_sparse:
            raise TypeError("W must be a sparse tensor.")
        if W.layout != torch.sparse_coo:
            raise TypeError("W must have sparse COO layout.")
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must have shape (n, n).")
        if W.dtype != torch.float32:
            raise TypeError("W must have dtype torch.float32.")

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
        """Simulate equilibrium via differentiable Picard + Newton iteration."""
        self._validate_forward_inputs(W=W, X=X)

        lam = self.lambda_max * torch.sigmoid(self.raw_lambda)
        mu = torch.exp(self.log_mu)
        if self._fixed_r is not None:
            r = self._fixed_r
        else:
            r = torch.exp(self.log_r)
        if self._fixed_sigma_sq is not None:
            sigma = self._fixed_sigma_sq ** 0.5
        else:
            sigma = torch.sqrt(torch.exp(self.log_sigma_sq))

        eps = sigma * torch.randn_like(X)

        if self._fixed_r is not None:
            z_clamp_bound: float | Tensor = 50.0 / self._fixed_r
        else:
            z_clamp_bound = 50.0 / r

        Y = torch.zeros_like(X)
        picard_iters_used = self.picard_max
        newton_max_used = 0

        for t in range(self.picard_max):
            WY = torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)
            b = lam * WY + self.gamma * X + eps

            z = (b / (1.0 + lam)).clamp(-z_clamp_bound, z_clamp_bound)
            newton_iters_used = self.newton_max

            for s in range(self.newton_max):
                exp_neg_rz = torch.exp(-r * z)
                f_val = (1.0 + lam) * z - mu * r * exp_neg_rz - b
                f_prime = (1.0 + lam) + mu * r * r * exp_neg_rz
                delta = f_val / f_prime
                z = z - delta

                if not self.fixed_iterations:
                    max_delta = delta.detach().abs().max().item()
                    if max_delta < self.newton_tol:
                        newton_iters_used = s + 1
                        break

            newton_max_used = max(newton_max_used, newton_iters_used)
            Y_next = z

            if not self.fixed_iterations:
                picard_delta = (Y_next - Y).detach().abs().max().item()
                Y = Y_next
                if picard_delta < self.picard_tol:
                    picard_iters_used = t + 1
                    break
            else:
                Y = Y_next

        self.last_picard_iterations = picard_iters_used
        self.last_newton_max_iters = newton_max_used
        return Y

    def get_params(self) -> dict[str, float]:
        """Return constrained scalar parameters as detached Python floats."""
        with torch.no_grad():
            lambda_ = self.lambda_max * torch.sigmoid(self.raw_lambda)
            mu = torch.exp(self.log_mu)
            if self._fixed_sigma_sq is not None:
                sigma_sq = self._fixed_sigma_sq
            else:
                sigma_sq = float(torch.exp(self.log_sigma_sq).item())
            if self._fixed_r is not None:
                r_val = self._fixed_r
            else:
                r_val = float(torch.exp(self.log_r).item())

            return {
                "gamma": float(self.gamma.item()),
                "lambda_": float(lambda_.item()),
                "mu": float(mu.item()),
                "r": float(r_val),
                "sigma_sq": float(sigma_sq),
            }

    @property
    def contraction_rate(self) -> float:
        """Current contraction rate rho = lambda/(1+lambda)."""
        with torch.no_grad():
            lambda_ = self.lambda_max * torch.sigmoid(self.raw_lambda)
            rho = lambda_ / (1.0 + lambda_)
            return float(rho.item())
