"""Tests for :class:`RootedMPNNDiscriminator` — the adaptive test function (C6).

These guard the three C6 properties of Illichmann & Zacchia (2026):

* **C6(i) clip.** The default discriminator's score ``D = sigmoid(logit)`` is
  bounded inside ``(eta, 1 - eta)`` with ``eta = sigmoid(-logit_clip)``, so the
  per-object loss ``|log D|`` is bounded; the soft clamp keeps ``logit = 0 -> 1/2``
  (loss optima unchanged) and a non-zero gradient for finite activations.
* **C6 single object.** The root logit depends only on the single rooted ego
  object, not on which other ego objects share the minibatch — the regression
  guard for the old ``BatchNorm`` cross-object leak (it would fail under
  ``BatchNorm`` and passes under ``LayerNorm``).
* **C6(i) Lipschitz trunk.** Every linear map (GIN-MLP and head) is spectrally
  normalised, and no ``BatchNorm`` survives.

Batches are built with the same ``k_hop_subgraph`` + PyG ``Data``/``Batch`` idiom
used across the test-suite (see ``tests/test_core_ego_features.py``).
"""

from __future__ import annotations

import math

import networkx as nx
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx, k_hop_subgraph, to_undirected

from adversarial_networks.discriminator import RootedMPNNDiscriminator


# --------------------------------------------------------------------------- utils
def _path_graph_edge_index(num_nodes: int) -> torch.Tensor:
    """Edge index of an undirected path graph on ``num_nodes`` nodes."""
    graph = nx.path_graph(num_nodes)
    data = from_networkx(graph)
    return to_undirected(data.edge_index, num_nodes=num_nodes)


def _ego_data(root: int, edge_index: torch.Tensor, num_nodes: int, k: int, *, scale: float = 1.0) -> Data:
    """Build a single rooted ego ``Data`` ``[X_tilde, Y_tilde, root_marker]``.

    Mirrors :func:`core.ego_features.extract_ego_batch`: induce the relabelled
    ``k``-hop subgraph, mark the root channel, and attach random covariate/outcome
    channels (``scale`` lets a test drive large pre-clip activations).
    """
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=root,
        num_hops=k,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    m = int(subset.shape[0])
    x_tilde = torch.randn(m) * scale
    y_tilde = torch.randn(m) * scale
    root_marker = torch.zeros(m)
    root_marker[int(mapping.item())] = 1.0
    features = torch.stack((x_tilde, y_tilde, root_marker), dim=1)
    return Data(x=features, edge_index=sub_edge_index)


def _batch_from(data_list: list[Data]) -> tuple[Batch, torch.Tensor]:
    """Concatenate ego ``Data`` into a PyG ``Batch`` and return ``(batch, root_indices)``.

    The root index is the per-graph ``ptr`` offset plus the local root position
    (here the root is the relabelled node carrying ``root_marker == 1``).
    """
    batch = Batch.from_data_list(data_list)
    ptr = batch.ptr[:-1]
    local_roots = torch.tensor(
        [int((d.x[:, 2] == 1.0).nonzero(as_tuple=True)[0].item()) for d in data_list],
        dtype=torch.long,
    )
    return batch, ptr + local_roots


# ------------------------------------------------------------------- C6(i): clip
def test_default_discriminator_scores_strictly_inside_eta_band() -> None:
    """Random inputs: D = sigmoid(logit) lies strictly within (eta, 1-eta)."""
    torch.manual_seed(0)
    disc = RootedMPNNDiscriminator(hidden_dim=16, num_layers=2)
    assert disc.logit_clip == 5.0
    eta = 1.0 / (1.0 + math.exp(disc.logit_clip))  # sigmoid(-c)

    n = 30
    edge_index = _path_graph_edge_index(n)
    data_list = [_ego_data(r, edge_index, n, k=2) for r in range(0, n, 2)]
    batch, root_idx = _batch_from(data_list)

    disc.eval()
    with torch.no_grad():
        scores = torch.sigmoid(disc(batch.x, batch.edge_index, root_idx))

    assert float(scores.min()) > eta, f"min score {float(scores.min())} <= eta {eta}"
    assert float(scores.max()) < 1.0 - eta, f"max score {float(scores.max())} >= 1-eta {1.0 - eta}"


def test_large_pre_clip_logits_respect_eta_band() -> None:
    """Large *pre-clip* logits still produce D in [eta, 1-eta], and the clamp engages.

    The per-conv LayerNorm + spectral norm keep the representation O(1), so raw input
    scale never reaches the head as a large logit — the clip is the safety bound for a
    *trained* discriminator whose head weights have grown. We emulate that state by
    scaling the head, and compare against an otherwise-identical ``logit_clip=None``
    twin (same ``state_dict``): the unclipped twin produces logits beyond ``c`` while
    the clipped net keeps every logit within ``[-c, c]`` (so ``D in [eta, 1-eta]``)
    and demonstrably reduces the magnitude. This tests ``forward``'s clamp directly,
    with no brittle saturation threshold.
    """
    torch.manual_seed(2)
    clipped = RootedMPNNDiscriminator(hidden_dim=16, num_layers=2)
    unclipped = RootedMPNNDiscriminator(hidden_dim=16, num_layers=2, logit_clip=None)
    unclipped.load_state_dict(clipped.state_dict())  # identical weights, clamp the only diff
    c = clipped.logit_clip
    eta = 1.0 / (1.0 + math.exp(c))

    with torch.no_grad():  # grow the head so pre-clip logits exceed c
        for disc in (clipped, unclipped):
            for module in disc.head:
                if isinstance(module, nn.Linear):
                    module.parametrizations.weight.original.mul_(40.0)
                    if module.bias is not None:
                        module.bias.add_(15.0)

    n = 24
    edge_index = _path_graph_edge_index(n)
    data_list = [_ego_data(r, edge_index, n, k=2, scale=5.0) for r in range(0, n, 2)]
    batch, root_idx = _batch_from(data_list)

    clipped.eval()
    unclipped.eval()
    with torch.no_grad():
        logits_clipped = clipped(batch.x, batch.edge_index, root_idx)
        logits_unclipped = unclipped(batch.x, batch.edge_index, root_idx)
        scores = torch.sigmoid(logits_clipped)

    # The unclipped twin genuinely exceeds c (so the clamp has real work to do)...
    assert float(logits_unclipped.abs().max()) > c
    # ...and the clipped net keeps every logit within the band, so D in [eta, 1-eta].
    assert float(logits_clipped.abs().max()) <= c + 1e-6
    assert float(scores.min()) >= eta - 1e-6
    assert float(scores.max()) <= (1.0 - eta) + 1e-6
    # The clamp strictly reduced the magnitude where the logit was large.
    assert bool((logits_clipped.abs() < logits_unclipped.abs() - 1e-4).any())


def test_logit_clip_none_disables_clip_and_is_validated() -> None:
    """logit_clip=None opts out of the C6(i) bound; a non-positive value is rejected."""
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=None)
    assert disc.logit_clip is None

    n = 18
    edge_index = _path_graph_edge_index(n)
    data_list = [_ego_data(r, edge_index, n, k=2, scale=50.0) for r in range(0, n, 3)]
    batch, root_idx = _batch_from(data_list)
    disc.eval()
    with torch.no_grad():
        logits = disc(batch.x, batch.edge_index, root_idx)
    # Without the clamp the logits are free to exceed the default half-width.
    assert torch.isfinite(logits).all()

    for bad in (0.0, -1.0):
        try:
            RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=bad)
        except ValueError:
            pass
        else:  # pragma: no cover - guard against a silent contract regression
            raise AssertionError(f"logit_clip={bad} should have raised ValueError.")


# ------------------------------------------------- C6 single-object (the leak guard)
def test_root_logit_is_invariant_to_batch_composition() -> None:
    """The single-object property: a root's logit is unchanged by batch neighbours.

    Build the SAME ego object once alone and once concatenated with several OTHER
    ego objects; the discriminator's logit for that object's root must be equal
    (tight tol) regardless of batch composition, in both train and eval mode. This
    holds under LayerNorm (per-node statistic) and FAILS under BatchNorm (the score
    would depend on the rest of the minibatch) — the regression guard for the leak.
    """
    torch.manual_seed(7)
    disc = RootedMPNNDiscriminator(hidden_dim=16, num_layers=2)

    n = 40
    edge_index = _path_graph_edge_index(n)

    target_root = 5
    # Fix the target object's features so it is bit-identical in both batches.
    torch.manual_seed(123)
    target = _ego_data(target_root, edge_index, n, k=2)

    others = [_ego_data(r, edge_index, n, k=2) for r in (12, 20, 27, 33)]

    batch_alone, idx_alone = _batch_from([target])
    batch_mixed, idx_mixed = _batch_from([target, *others])

    for mode in ("eval", "train"):
        getattr(disc, mode)()
        with torch.no_grad():
            logit_alone = disc(batch_alone.x, batch_alone.edge_index, idx_alone)[0]
            # The target is the FIRST graph in the mixed batch, so its root is idx 0.
            logit_mixed = disc(batch_mixed.x, batch_mixed.edge_index, idx_mixed)[0]
        assert torch.allclose(logit_alone, logit_mixed, atol=1e-5, rtol=0.0), (
            f"[{mode}] root logit changed with batch composition: "
            f"alone={float(logit_alone):.8f} mixed={float(logit_mixed):.8f} "
            "(LayerNorm should make D a function of the single ego object; "
            "BatchNorm would leak the batch)."
        )


# ------------------------------------------- C6 receptive field (footnote-33 depth)
def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Undirected path-graph edge index ``0-1-2-...-(num_nodes-1)`` (exact hop distances)."""
    src: list[int] = []
    dst: list[int] = []
    for i in range(num_nodes - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    return torch.tensor([src, dst], dtype=torch.long)


def _path_features(num_nodes: int, root: int, *, perturb_node: int | None = None) -> torch.Tensor:
    """Node rows ``[X_tilde, Y_tilde, root_marker]`` on the path; optionally perturb one
    node's ``Y_tilde`` channel to a distinctive value."""
    x = torch.full((num_nodes, 3), 0.1)
    x[root, 2] = 1.0  # root marker
    if perturb_node is not None:
        x[perturb_node, 1] = 5.0  # distinctive feature on the probed node's Y_tilde
    return x


def test_receptive_field_reaches_radius_k_only_with_k_layers() -> None:
    """Footnote-33 depth property (behavioural, not the numeric ``num_layers >= k`` guard).

    On a path ``0-1-2-3-...`` rooted at node 0, a node at *exactly* distance ``k`` from the
    root is reachable in ``k`` message-passing rounds but not in ``k - 1``. So a
    discriminator with ``num_layers == k`` must produce a DIFFERENT root logit when that
    distance-``k`` node's feature changes (information reaches the root), while
    ``num_layers == k - 1`` must produce the IDENTICAL root logit (the distance-``k`` node
    is outside the receptive field). The same path / root / feature is used for both depths,
    so the only varying factor is ``num_layers``. This would pass the numeric guard yet
    fail here if a GINConv/aggregation/depth regression broke the true receptive field.
    """
    k = 3
    num_nodes = 8  # path long enough that a distance-k node (node 3) exists, with slack
    root = 0
    dist_k_node = k  # on a path rooted at 0, node index == hop distance from the root
    edge_index = _path_edge_index(num_nodes)
    root_indices = torch.tensor([root], dtype=torch.long)

    base = _path_features(num_nodes, root)
    perturbed = _path_features(num_nodes, root, perturb_node=dist_k_node)

    # num_layers == k - 1: the distance-k node is OUTSIDE the receptive field -> no effect.
    torch.manual_seed(2)
    shallow = RootedMPNNDiscriminator(hidden_dim=16, num_layers=k - 1, logit_clip=5.0)
    shallow.eval()
    with torch.no_grad():
        logit_base_shallow = shallow(base, edge_index, root_indices)[0]
        logit_pert_shallow = shallow(perturbed, edge_index, root_indices)[0]
    assert float((logit_pert_shallow - logit_base_shallow).abs()) == 0.0, (
        "a discriminator with num_layers < k must NOT see a node at distance k from the "
        "root (it is outside the receptive field), so the root logit must be unchanged."
    )

    # num_layers == k: the distance-k node is INSIDE the receptive field -> the logit moves.
    torch.manual_seed(2)
    deep = RootedMPNNDiscriminator(hidden_dim=16, num_layers=k, logit_clip=5.0)
    deep.eval()
    with torch.no_grad():
        logit_base_deep = deep(base, edge_index, root_indices)[0]
        logit_pert_deep = deep(perturbed, edge_index, root_indices)[0]
    assert float((logit_pert_deep - logit_base_deep).abs()) > 1e-5, (
        "a discriminator with num_layers >= k must give the root embedding access to the "
        "full radius-k ball, so perturbing a distance-k node's feature must change the "
        "root logit (footnote 33)."
    )


# ----------------------------------------------------- soft-clamp mapping (direct)
def test_soft_clamp_maps_zero_to_zero_and_bounds_magnitude() -> None:
    """Direct unit test of ``c * tanh(logit / c)``: clamp(0)=0 and |clamp(x)|<c.

    For finite, non-saturating ``x`` the bound is *strict* (``< c``) and the slope is
    preserved (non-zero gradient), unlike a hard clamp. For extreme ``x`` float
    ``tanh`` saturates to exactly ``+/-1`` so the closed bound ``<= c`` is the
    guarantee there; both are asserted.
    """
    c = 5.0
    # Non-saturating regime: strict bound + preserved gradient.
    x = torch.tensor([0.0, 0.5, -0.5, 3.0, -3.0, 20.0, -20.0], requires_grad=True)
    clamped = c * torch.tanh(x / c)

    assert float(clamped[0].detach()) == 0.0, "clamp(0) must be exactly 0 (preserves D=1/2)."
    assert torch.all(clamped.detach().abs() < c), "finite logits must map strictly inside (-c, c)."

    clamped.sum().backward()
    assert torch.all(x.grad > 0.0), "soft clamp must keep a non-zero gradient everywhere finite."

    # Extreme regime: float tanh saturates -> the closed bound |clamp| <= c holds.
    x_big = torch.tensor([1e6, -1e6, 1e30, -1e30])
    clamped_big = c * torch.tanh(x_big / c)
    assert torch.all(clamped_big.abs() <= c), "extreme logits must still respect |clamp| <= c."
    assert float(c * math.tanh(0.0 / c)) == 0.0


# --------------------------------------------------- architecture composition (C6)
def test_uses_layernorm_not_batchnorm() -> None:
    """No BatchNorm survives; per-conv normalisation is LayerNorm."""
    disc = RootedMPNNDiscriminator(hidden_dim=12, num_layers=3)
    batchnorms = [m for m in disc.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    layernorms = [m for m in disc.norms if isinstance(m, nn.LayerNorm)]
    assert batchnorms == [], "BatchNorm leaks batch composition into D; it must be gone."
    assert len(layernorms) == disc.num_layers
    assert not hasattr(disc, "bns"), "the old BatchNorm ModuleList must be removed."


def test_spectral_norm_on_trunk_and_head() -> None:
    """Every linear map (GIN-MLP + head) is spectrally normalised (C6(i) trunk)."""
    disc = RootedMPNNDiscriminator(hidden_dim=12, num_layers=2)

    conv_linears = [m for conv in disc.convs for m in conv.nn if isinstance(m, nn.Linear)]
    head_linears = [m for m in disc.head if isinstance(m, nn.Linear)]
    assert len(conv_linears) == 2 * disc.num_layers  # two linears per GIN MLP
    assert len(head_linears) == 2

    for linear in conv_linears + head_linears:
        assert parametrize.is_parametrized(linear, "weight"), (
            "every linear (trunk and head) must carry a spectral-norm parametrization."
        )
