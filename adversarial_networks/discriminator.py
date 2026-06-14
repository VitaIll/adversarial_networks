"""Root-aware MPNN discriminator for rooted ego-subgraph classification.

The discriminator classifies rooted k-ego subgraphs as real (observed) or fake
(simulated). Message-passing depth is configurable so receptive-field radius can
be matched to the subgraph radius used by the experiment.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import GINConv


class RootedMPNNDiscriminator(nn.Module):
    """Root-aware discriminator over rooted attributed ego-subgraphs.

    Each node feature row is `[X_tilde, Y_tilde, root_marker]`.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        logit_clip: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if logit_clip is not None and logit_clip <= 0:
            raise ValueError("logit_clip must be positive when provided.")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.logit_clip = float(logit_clip) if logit_clip is not None else None

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_dim = 3
        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim, track_running_stats=False))
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
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("x must have shape (num_nodes, 3).")
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
        """Return per-root logits for real/fake classification."""
        self._validate_inputs(x=x, edge_index=edge_index, root_indices=root_indices)

        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = F.relu(bn(h))

        root_h = h.index_select(0, root_indices)
        logits = self.head(root_h).squeeze(-1)
        if self.logit_clip is not None:
            logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
        return logits
