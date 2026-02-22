"""Structural Causal Model (SCM) Generator for Linear-in-Means Equilibrium.

This module implements the generator component of the adversarial structural estimator,
which simulates equilibrium outcomes from a linear-in-means social interaction model
using Picard iteration with automatic differentiation.

The model: Y = β·W·Y + X·γ + ε, where ε ~ N(0, σ²I)
Equilibrium: Y = (I - β·W)^(-1)·(X·γ + ε) for |β| < 1

References:
    - Kaji, Manresa & Pouliot (2023). "An Adversarial Approach to Structural Estimation."
      Econometrica, 91(6), 2041-2063.
    - Bramoullé, Djebbari & Fortin (2009). "Identification of peer effects through
      social networks." Journal of Econometrics, 150(1), 41-55.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SCMGenerator(nn.Module):
    """Structural simulator for linear-in-means equilibrium with differentiable Picard iteration.

    This generator implements the structural causal model:
        Y = β·W·Y + X·γ + ε, with ε_i ~ N(0, σ²)

    where:
        - β: peer effect parameter (social multiplier strength)
        - W: row-stochastic network weight matrix
        - γ: exogenous effect of covariate X on outcome Y
        - σ²: variance of idiosyncratic shocks

    The equilibrium Y is computed via Picard iteration:
        Y^(t+1) = β·W·Y^(t) + X·γ + ε, starting from Y^(0) = 0

    Parameters are constrained via reparameterization to ensure stability:
        - β = β_cap · tanh(raw_β) enforces |β| < β_cap < 1 (contraction)
        - σ² = exp(log_σ²) enforces σ² > 0 (positive variance)
        - γ ∈ ℝ (unconstrained)

    Args:
        beta_cap: Contraction cap for β, must satisfy 0 < β_cap < 1.
            This ensures Picard iteration converges for any optimizer step.
        picard_tol: Convergence threshold for early stopping. Iteration stops when
            max|Y^(t+1) - Y^(t)| < picard_tol. Must be positive.
        picard_max: Maximum number of Picard iterations. Must be positive.
        init_beta: Initial value for constrained β parameter.
            Must satisfy |init_beta| < beta_cap.
        init_gamma: Initial value for exogenous effect γ.
        init_log_sigma_sq: Initial value for log(σ²). The variance σ² = exp(init_log_sigma_sq).

    Attributes:
        beta_cap: Stored contraction cap (float).
        picard_tol: Stored convergence tolerance (float).
        picard_max: Stored max iterations (int).
        raw_beta: Learnable unconstrained parameter for β (nn.Parameter).
        gamma: Learnable parameter for exogenous effect (nn.Parameter).
        log_sigma_sq: Learnable parameter for log(σ²) (nn.Parameter).
        last_picard_iterations: Number of iterations used in most recent forward pass (int).

    Raises:
        ValueError: If beta_cap, picard_tol, picard_max, or init_beta violate constraints.

    Example:
        >>> import torch
        >>> from torch_geometric.utils import from_networkx
        >>> import networkx as nx
        >>> from src.utils import build_row_stochastic_W
        >>>
        >>> # Create a small graph
        >>> G = nx.karate_club_graph()
        >>> n = G.number_of_nodes()
        >>> edge_index = from_networkx(G).edge_index
        >>> W = build_row_stochastic_W(edge_index, num_nodes=n)
        >>>
        >>> # Create generator with true parameters
        >>> generator = SCMGenerator(
        ...     beta_cap=0.8,
        ...     picard_tol=1e-6,
        ...     picard_max=100,
        ...     init_beta=0.4,
        ...     init_gamma=1.5,
        ...     init_log_sigma_sq=0.0,  # σ² = 1.0
        ... )
        >>>
        >>> # Generate covariate and simulate equilibrium
        >>> X = torch.randn(n)
        >>> Y_sim = generator(W, X)
        >>> print(f"Converged in {generator.last_picard_iterations} iterations")
        >>>
        >>> # Access constrained parameters
        >>> params = generator.get_params()
        >>> print(f"β={params['beta']:.3f}, γ={params['gamma']:.3f}, σ²={params['sigma_sq']:.3f}")

    Notes:
        - The forward pass is fully differentiable via PyTorch autograd. Gradients
          backpropagate through all Picard iterations to the leaf parameters.
        - Memory cost: O(T·n) where T is the number of iterations and n is the number
          of nodes. For n=1000 and T=100, this is ~0.4 MB in float32.
        - Truncation error: O(|β|^T) where T is the number of iterations. With
          |β| = 0.4 and T = 100, error is below float32 precision.
        - The reparameterization trick (ε = σ·z, z ~ N(0,I)) ensures gradients flow
          to σ via the MulBackward operation.
        - No `.detach()` is used inside the Picard loop to preserve gradient flow.
          Only the convergence check uses `.detach()` to avoid recording it on tape.
    """

    def __init__(
        self,
        beta_cap: float = 0.8,
        picard_tol: float = 1e-6,
        picard_max: int = 100,
        init_beta: float = 0.0,
        init_gamma: float = 0.0,
        init_log_sigma_sq: float = 0.0,
    ) -> None:
        super().__init__()

        # Validate hyperparameters
        if not (0.0 < beta_cap < 1.0):
            raise ValueError("beta_cap must satisfy 0 < beta_cap < 1.")
        if not (picard_tol > 0.0):
            raise ValueError("picard_tol must be strictly positive.")
        if picard_max <= 0:
            raise ValueError("picard_max must be positive.")
        if abs(init_beta) >= beta_cap:
            raise ValueError("init_beta must satisfy abs(init_beta) < beta_cap.")

        self.beta_cap = float(beta_cap)
        self.picard_tol = float(picard_tol)
        self.picard_max = int(picard_max)

        # Initialize learnable parameters with reparameterizations
        # β = β_cap · tanh(raw_β) ∈ (-β_cap, β_cap)
        scaled_beta = torch.tensor(init_beta / self.beta_cap, dtype=torch.float32)
        self.raw_beta = nn.Parameter(torch.atanh(scaled_beta))

        # γ ∈ ℝ (unconstrained)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma), dtype=torch.float32))

        # σ² = exp(log_σ²) ∈ (0, ∞)
        self.log_sigma_sq = nn.Parameter(
            torch.tensor(float(init_log_sigma_sq), dtype=torch.float32)
        )

        # Track iterations for diagnostics
        self.last_picard_iterations: int = 0

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
        """Simulate equilibrium outcome Y via differentiable Picard iteration.

        Computes the equilibrium:
            Y = β·W·Y + X·γ + ε

        using Picard iteration with fresh noise ε ~ N(0, σ²I). All operations
        are recorded by autograd for gradient backpropagation.

        Args:
            W: Sparse COO tensor of shape (n, n), dtype float32, representing the
                row-stochastic network weight matrix. Must be on same device as X.
            X: Dense tensor of shape (n,), floating dtype, representing node covariates.
                Must be on same device as W.

        Returns:
            Y_sim: Simulated equilibrium outcome, shape (n,), floating dtype, same
                device as X. Gradients flow to all learnable parameters (raw_beta,
                gamma, log_sigma_sq).

        Raises:
            TypeError: If input types or dtypes are invalid.
            ValueError: If shapes or devices are incompatible.

        Notes:
            - The method samples fresh noise ε = σ·z where z ~ N(0, I) using the
              reparameterization trick for gradient flow.
            - Iteration: Y^(t+1) = β·W·Y^(t) + X·γ + ε, starting from Y^(0) = 0.
            - Early stopping: when max|Y^(t+1) - Y^(t)| < picard_tol.
            - The convergence check uses `.detach()` only for the stopping criterion;
              Y itself remains on the autograd tape throughout.
            - `last_picard_iterations` is updated with the number of iterations used.
        """
        self._validate_forward_inputs(W=W, X=X)

        # Apply constrained reparameterizations
        beta = self.beta_cap * torch.tanh(self.raw_beta)
        sigma_sq = torch.exp(self.log_sigma_sq)
        sigma = torch.sqrt(sigma_sq)

        # Sample noise using reparameterization trick
        eps = sigma * torch.randn_like(X)

        # Compute base term (exogenous + shock)
        base = X * self.gamma + eps

        # Picard iteration: Y^(t+1) = β·W·Y^(t) + base
        Y = torch.zeros_like(X)
        iterations_used = self.picard_max

        for step in range(self.picard_max):
            # Sparse matrix-vector product: W·Y
            WY = torch.sparse.mm(W, Y.unsqueeze(-1)).squeeze(-1)
            Y_next = beta * WY + base

            # Check convergence (off-tape, for stopping only)
            max_delta = (Y_next - Y).detach().abs().max().item()
            Y = Y_next

            if max_delta < self.picard_tol:
                iterations_used = step + 1
                break

        self.last_picard_iterations = iterations_used
        return Y

    def get_params(self) -> dict[str, float]:
        """Return constrained scalar parameters as detached Python floats.

        Returns:
            Dictionary with keys "beta", "gamma", "sigma_sq" mapping to
            constrained parameter values as Python floats (no gradient).

        Example:
            >>> params = generator.get_params()
            >>> print(f"Current parameters: β={params['beta']:.4f}, "
            ...       f"γ={params['gamma']:.4f}, σ²={params['sigma_sq']:.4f}")
        """
        with torch.no_grad():
            beta = self.beta_cap * torch.tanh(self.raw_beta)
            sigma_sq = torch.exp(self.log_sigma_sq)
            return {
                "beta": float(beta.item()),
                "gamma": float(self.gamma.item()),
                "sigma_sq": float(sigma_sq.item()),
            }
