"""Root-aware MPNN discriminator for rooted ego-subgraph classification.

The discriminator is the paper's adaptive test function ``D_phi`` (Illichmann &
Zacchia 2026). It classifies rooted ``k``-ego subgraphs as real (observed) or fake
(simulated). Message-passing depth is configurable so the receptive-field radius
can be matched to the subgraph radius used by the experiment.

Condition C6 requires the test function to (i) be clipped to a *fixed*
``[eta, 1 - eta]`` band so the per-object loss ``|log D| <= |log eta|`` is bounded
(the concentration / consistency argument and the FiniteMoment theorem, Thm 13);
(ii) be ``L_D``-Lipschitz in the ego metric; and (iii) be a function of a *single*
ego object ``S_{n,k} -> [eta, 1 - eta]`` — its output for a root must not depend on
which other ego objects share the minibatch. This module is built to honour all
three:

* **Clip (C6(i)).** A soft, gradient-preserving logit bound
  ``logit = c * tanh(logit / c)`` with ``c = logit_clip`` maps the score into the
  open interval ``D = sigmoid(logit) in (sigmoid(-c), sigmoid(c)) = (eta, 1 - eta)``
  for every finite pre-clip activation, with ``eta = sigmoid(-c)``. ``logit = 0``
  still maps to ``D = 1/2`` so the population loss optima (``2 log 2`` / ``log 2``)
  are unchanged. The clip is on by default; passing ``logit_clip=None`` disables it
  and drops the C6(i) guarantee (``D`` then ranges over the open ``(0, 1)``).
* **Per-object normalisation (C6 single-object).** Normalisation is
  :class:`~torch.nn.LayerNorm` over each node's hidden channels — a per-node
  statistic with no cross-sample / cross-graph coupling, identical in train and
  eval. ``D`` is therefore a function of the single ego object alone; its root
  logit is invariant to minibatch composition. (A ``BatchNorm`` here would
  normalise over the whole minibatch and make ``D`` depend on the batch, violating
  the single-object requirement — the estimator runs real and fake through
  separate forward passes, so a batch statistic would leak batch composition into
  the score.)
* **Lipschitz trunk (C6(i)).** Spectral normalisation wraps every linear map — the
  two ``nn.Linear`` layers inside each GIN message-passing MLP *and* the two head
  layers — so each parametric map has a bounded spectral norm. The GIN additive
  aggregation contributes a *degree-dependent* gain (a root of degree ``d`` sums up
  to ``d`` neighbour messages). Under the maintained admissibility assumptions —
  bounded degree (A1) / finite first-moment branching (G2) — the per-node neighbour
  count is uniformly bounded by the maximum degree ``Delta``, so that aggregation
  gain is itself *uniformly* bounded; combined with the spectral-normed linears and
  the bounded (clipped + per-object LayerNorm-normalised) representations, the
  composed test function ``D_phi`` is ``L_D``-Lipschitz in the ego metric with a
  finite, ``n``-uniform constant — i.e. C6(i) **holds** on the admissible
  bounded-degree class. The constant *degrades* with the maximum degree: the bound
  scales with ``Delta``, so very high-degree hubs inflate ``L_D`` (and a graph
  ensemble with an unbounded second degree moment — outside A1/G2 — forfeits the
  uniform constant). A degree-normalised (mean) aggregation would make the gain
  degree-independent; the additive GIN form is retained for its tested numerics, the
  uniform constant being supplied by the bounded-degree assumption rather than by the
  architecture alone.

References:
    Illichmann & Zacchia (2026), condition C6 and Theorem 13.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import GINConv


class RootedMPNNDiscriminator(nn.Module):
    """Root-aware discriminator over rooted attributed ego-subgraphs.

    Each node feature row is ``[X_tilde (d_x cols), Y_tilde, root_marker]``, so the
    input width is ``in_dim = d_x + 2`` (``3`` for the scalar-covariate default
    ``d_x = 1``). The forward pass returns one logit per root; ``sigmoid`` of that
    logit is the test function ``D_phi`` evaluated on the rooted ego object.

    Args:
        hidden_dim: Width of the GIN hidden channels and the head (must be > 0).
        num_layers: Number of GIN message-passing layers (must be > 0); the ego
            radius ``k`` requires ``num_layers >= k`` so the receptive field covers
            the ego neighbourhood (enforced by the estimator, not here).
        in_dim: Width of the node feature row (must be > 0); equals ``d_x + 2`` for a
            ``d_x``-column covariate. Defaults to ``3`` (the scalar-covariate case).
        logit_clip: Half-width ``c`` of the soft logit bound (must be > 0 when
            given). The default ``5.0`` gives ``eta = sigmoid(-5) ~ 0.0067`` and
            ``|log eta| ~ 5``, satisfying C6(i). Pass ``None`` to disable clipping;
            this drops the C6(i) guarantee and optimises the ``eta = 0`` criterion,
            for which the paper provides no consistency result.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        in_dim: int = 3,
        logit_clip: float | None = 5.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if in_dim <= 0:
            raise ValueError("in_dim must be positive (equals d_x + 2 for a d_x-column covariate).")
        if logit_clip is not None and logit_clip <= 0:
            raise ValueError("logit_clip must be positive when provided.")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.in_dim = int(in_dim)
        self.logit_clip = float(logit_clip) if logit_clip is not None else None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = self.in_dim
        for _ in range(self.num_layers):
            # Spectral-normalise both linears of the GIN MLP so the message-passing
            # trunk (not only the head) is Lipschitz-constrained (C6(i)).
            mlp = nn.Sequential(
                spectral_norm(nn.Linear(in_dim, self.hidden_dim)),
                nn.ReLU(),
                spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            # LayerNorm normalises each node's hidden vector across channels: a
            # per-node statistic with no cross-sample / cross-graph dependence and
            # identical behaviour in train and eval, so D stays a function of the
            # single ego object (the C6 single-object requirement).
            self.norms.append(nn.LayerNorm(self.hidden_dim))
            in_dim = self.hidden_dim

        self.head = nn.Sequential(
            spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(self.hidden_dim, 1)),
        )

    def _validate_inputs(
        self, x: Tensor, edge_index: Tensor, root_indices: Tensor
    ) -> None:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(
                f"x must have shape (num_nodes, in_dim={self.in_dim}); the node feature "
                f"row is [X_tilde (d_x), Y_tilde, root_marker] so in_dim = d_x + 2."
            )
        if not torch.is_floating_point(x):
            raise TypeError("x must have a floating dtype.")

        if not isinstance(edge_index, Tensor):
            raise TypeError("edge_index must be a torch.Tensor.")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")

        if not isinstance(root_indices, Tensor):
            raise TypeError("root_indices must be a torch.Tensor.")
        if root_indices.dtype != torch.long:
            raise TypeError("root_indices must have dtype torch.long.")
        if root_indices.ndim != 1:
            raise ValueError("root_indices must have shape (num_subgraphs,).")
        if root_indices.numel() == 0:
            raise ValueError("root_indices cannot be empty.")

        if int(root_indices.min().item()) < 0 or int(root_indices.max().item()) >= x.shape[0]:
            raise ValueError("root_indices contain values outside valid node range.")

        if x.device != edge_index.device or x.device != root_indices.device:
            raise ValueError("x, edge_index, and root_indices must be on the same device.")

    def forward(self, x: Tensor, edge_index: Tensor, root_indices: Tensor) -> Tensor:
        """Return per-root logits for real/fake classification.

        With ``logit_clip`` set (the default), each returned logit lies in the open
        interval ``(-logit_clip, logit_clip)`` for finite inputs, so
        ``D = sigmoid(logit)`` lies in ``(eta, 1 - eta)`` with ``eta =
        sigmoid(-logit_clip)`` (the C6(i) bound). The root logit depends only on the
        single rooted ego object, not on the rest of the minibatch.
        """
        self._validate_inputs(x=x, edge_index=edge_index, root_indices=root_indices)

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = F.relu(norm(h))

        root_h = h.index_select(0, root_indices)
        logits = self.head(root_h).squeeze(-1)
        if self.logit_clip is not None:
            # Soft, gradient-preserving bound. Unlike a hard clamp (which zeroes the
            # gradient for any |logit| > c), c*tanh(logit/c) keeps a non-zero slope
            # for finite pre-clip activations. It maps logit -> (-c, c), hence
            # D = sigmoid(logit) -> (sigmoid(-c), sigmoid(c)) = (eta, 1 - eta), and
            # fixes logit=0 -> 0 so D=1/2 (the loss optima 2log2 / log2) is preserved.
            logits = self.logit_clip * torch.tanh(logits / self.logit_clip)
        return logits
