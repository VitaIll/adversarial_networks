"""Deterministic tests for configurable root sampling modes."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, k_hop_subgraph, to_undirected

from adversarial_networks import RootedMPNNDiscriminator
from adversarial_networks.core.ego_features import extract_ego_batch
from adversarial_networks.core.graph import (
    adjacency_lists_from_edge_index as build_adjacency_from_edge_index,
)
from adversarial_networks.core.graph import (
    row_stochastic_weights as build_row_stochastic_W,
)
from adversarial_networks.core.neighborhoods import (
    greedy_pack_best_from_permutations,
    greedy_pack_once_from_permutation,
    precompute_balls,
)
from adversarial_networks.generators import LinearInMeansGenerator as SCMGenerator
from adversarial_networks.sampling import RootSampler, sample_roots_tensor


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    """Convert a NetworkX graph to undirected edge_index."""
    data = from_networkx(graph)
    return to_undirected(data.edge_index, num_nodes=graph.number_of_nodes())


def _pairwise_distances_ok(graph: nx.Graph, roots: np.ndarray, radius: int) -> bool:
    """Check dist(u, v) > radius for all distinct selected roots."""
    all_dists = dict(nx.all_pairs_shortest_path_length(graph))
    root_list = [int(node) for node in roots.tolist()]
    for idx, u in enumerate(root_list):
        for v in root_list[idx + 1 :]:
            if all_dists[u][v] <= radius:
                return False
    return True


def test_disjoint_modes_enforce_pairwise_separation() -> None:
    """All disjoint variants must satisfy dist(u, v) > r for selected roots."""
    graphs = [
        nx.path_graph(20),
        nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4), ordering="sorted"),
        nx.connected_watts_strogatz_graph(18, 4, 0.2, seed=7),
    ]
    exclusion_r = 2

    for graph in graphs:
        num_nodes = graph.number_of_nodes()
        edge_index = _edge_index_from_graph(graph)
        adjacency = build_adjacency_from_edge_index(edge_index=edge_index, num_nodes=num_nodes)

        samplers = [
            RootSampler(
                num_nodes=num_nodes,
                mode="disjoint_once",
                exclusion_r=exclusion_r,
                adjacency=adjacency,
                rng=np.random.default_rng(101),
            ),
            RootSampler(
                num_nodes=num_nodes,
                mode="disjoint_best_of_k",
                exclusion_r=exclusion_r,
                disjoint_restarts_k=3,
                adjacency=adjacency,
                rng=np.random.default_rng(202),
            ),
            RootSampler(
                num_nodes=num_nodes,
                mode="disjoint_relax",
                exclusion_r=exclusion_r,
                disjoint_restarts_k=3,
                disjoint_min_batch=1,
                disjoint_relax_sequence=(exclusion_r,),
                disjoint_fallback="raise",
                adjacency=adjacency,
                rng=np.random.default_rng(303),
            ),
        ]

        for sampler in samplers:
            result = sampler.sample(batch_size=8)
            assert _pairwise_distances_ok(graph, result.roots, exclusion_r), (
                f"Sampler mode={sampler.mode} violated dist(u, v) > {exclusion_r} "
                f"on graph with n={num_nodes}."
            )


def test_precompute_balls_matches_path_graph_expected_nodes() -> None:
    """Closed balls B_r(v) on a path graph should match hand-computed nodes."""
    graph = nx.path_graph(10)
    edge_index = _edge_index_from_graph(graph)
    adjacency = build_adjacency_from_edge_index(edge_index=edge_index, num_nodes=10)
    balls = precompute_balls(adjacency=adjacency, radii=(1, 2, 3))

    assert set(balls[1][0].tolist()) == {0, 1}
    assert set(balls[2][0].tolist()) == {0, 1, 2}
    assert set(balls[2][4].tolist()) == {2, 3, 4, 5, 6}
    assert set(balls[3][4].tolist()) == {1, 2, 3, 4, 5, 6, 7}
    assert set(balls[2][9].tolist()) == {7, 8, 9}


def test_best_of_k_matches_max_over_given_permutations() -> None:
    """Best-of-k helper should return the largest set among tested permutations."""
    graph = nx.star_graph(5)
    edge_index = _edge_index_from_graph(graph)
    adjacency = build_adjacency_from_edge_index(edge_index=edge_index, num_nodes=6)
    balls = precompute_balls(adjacency=adjacency, radii=(1,))[1]

    perms = [
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        np.array([1, 2, 3, 4, 5, 0], dtype=np.int64),
        np.array([2, 0, 1, 3, 4, 5], dtype=np.int64),
    ]
    sizes = [
        greedy_pack_once_from_permutation(target_size=5, balls=balls, permutation=perm).size
        for perm in perms
    ]

    best, attempts = greedy_pack_best_from_permutations(
        target_size=5,
        balls=balls,
        permutations=perms,
    )
    assert best.size == max(sizes)
    assert attempts == 2, "Helper should early-exit after reaching target size."


def test_disjoint_relax_uses_weaker_radius_when_needed() -> None:
    """Relax mode should step down radius ladder when strict radius under-fills."""
    graph = nx.path_graph(12)
    edge_index = _edge_index_from_graph(graph)
    adjacency = build_adjacency_from_edge_index(edge_index=edge_index, num_nodes=12)

    sampler = RootSampler(
        num_nodes=12,
        mode="disjoint_relax",
        exclusion_r=3,
        disjoint_restarts_k=4,
        disjoint_min_batch=4,
        disjoint_relax_sequence=(3, 2),
        disjoint_fallback="raise",
        adjacency=adjacency,
        rng=np.random.default_rng(77),
    )
    result = sampler.sample(batch_size=6)

    assert result.radius_used == 2
    assert result.achieved_size >= 4
    assert result.mode == "disjoint_relax"
    assert _pairwise_distances_ok(graph, result.roots, radius=2)


def _run_training_smoke(mode: str) -> dict[str, int | str]:
    """Run a tiny GAN loop with paired roots for real/fake batches."""
    torch.manual_seed(99)
    np.random.seed(99)

    num_nodes = 36
    batch_size = 12
    n_disc = 2
    n_steps = 2
    k_hops = 2

    graph = nx.path_graph(num_nodes)
    edge_index = _edge_index_from_graph(graph)
    W = build_row_stochastic_W(edge_index=edge_index, num_nodes=num_nodes)
    X = torch.randn(num_nodes, dtype=torch.float32)

    true_generator = SCMGenerator(
        beta_cap=0.8,
        picard_tol=1e-6,
        picard_max=80,
        init_beta=0.35,
        init_gamma=1.2,
        init_log_sigma_sq=math.log(1.0),
    )
    with torch.no_grad():
        Y_obs = true_generator(W, X)

    norm_stats = {
        "mu_X": float(X.mean().item()),
        "sigma_X": float(X.std(unbiased=False).item()),
        "mu_Y": float(Y_obs.mean().item()),
        "sigma_Y": float(Y_obs.std(unbiased=False).item()),
    }

    ego_cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for root in range(num_nodes):
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=root,
            num_hops=k_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        ego_cache[root] = (subset, sub_edge_index, int(mapping.item()))

    generator = SCMGenerator(
        beta_cap=0.8,
        picard_tol=1e-6,
        picard_max=80,
        init_beta=0.0,
        init_gamma=0.0,
        init_log_sigma_sq=0.0,
    )
    discriminator = RootedMPNNDiscriminator(hidden_dim=16, num_layers=k_hops, logit_clip=10.0)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-3)

    rng = np.random.default_rng(123)
    if mode == "uniform":
        sampler = RootSampler(num_nodes=num_nodes, mode="uniform", rng=rng)
    elif mode == "disjoint_best_of_k":
        adjacency = build_adjacency_from_edge_index(edge_index=edge_index, num_nodes=num_nodes)
        sampler = RootSampler(
            num_nodes=num_nodes,
            mode="disjoint_best_of_k",
            exclusion_r=2,
            disjoint_restarts_k=3,
            adjacency=adjacency,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown test mode: {mode!r}")

    achieved_sizes: list[int] = []
    attempts_used: list[int] = []

    for _ in range(n_steps):
        with torch.no_grad():
            Y_sim_detached = generator(W, X)

        for _ in range(n_disc):
            roots, sample_info = sample_roots_tensor(
                sampler=sampler,
                batch_size=batch_size,
                device="cpu",
            )
            achieved_sizes.append(int(sample_info.achieved_size))
            attempts_used.append(int(sample_info.attempts_used))

            batch_real, root_real = extract_ego_batch(
                roots=roots,
                ego_cache=ego_cache,
                X=X,
                Y=Y_obs,
                norm_stats=norm_stats,
            )
            batch_fake, root_fake = extract_ego_batch(
                roots=roots,
                ego_cache=ego_cache,
                X=X,
                Y=Y_sim_detached,
                norm_stats=norm_stats,
            )

            opt_d.zero_grad(set_to_none=True)
            logits_real = discriminator(batch_real.x, batch_real.edge_index, root_real)
            logits_fake = discriminator(batch_fake.x, batch_fake.edge_index, root_fake)
            loss_d = F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()
            loss_d.backward()
            opt_d.step()

        roots_g, sample_info_g = sample_roots_tensor(
            sampler=sampler,
            batch_size=batch_size,
            device="cpu",
        )
        achieved_sizes.append(int(sample_info_g.achieved_size))
        attempts_used.append(int(sample_info_g.attempts_used))

        Y_sim = generator(W, X)
        batch_fake_g, root_fake_g = extract_ego_batch(
            roots=roots_g,
            ego_cache=ego_cache,
            X=X,
            Y=Y_sim,
            norm_stats=norm_stats,
        )
        opt_g.zero_grad(set_to_none=True)
        logits_fake_g = discriminator(batch_fake_g.x, batch_fake_g.edge_index, root_fake_g)
        loss_g = F.softplus(-logits_fake_g).mean()
        loss_g.backward()
        opt_g.step()

    log_line = (
        f"{mode} |S| min={min(achieved_sizes)} max={max(achieved_sizes)} "
        f"tries_max={max(attempts_used)}"
    )
    return {
        "calls": len(achieved_sizes),
        "min_achieved": int(min(achieved_sizes)),
        "max_achieved": int(max(achieved_sizes)),
        "tries_max": int(max(attempts_used)),
        "log_line": log_line,
    }


def test_training_smoke_uniform_and_disjoint_best_of_k() -> None:
    """Tiny end-to-end run should succeed for uniform and disjoint best-of-k."""
    uniform_stats = _run_training_smoke(mode="uniform")
    disjoint_stats = _run_training_smoke(mode="disjoint_best_of_k")

    assert uniform_stats["min_achieved"] == 12
    assert uniform_stats["max_achieved"] == 12
    assert disjoint_stats["calls"] > 0
    assert disjoint_stats["min_achieved"] > 0
    assert disjoint_stats["tries_max"] >= 1
    assert "|S|" in str(disjoint_stats["log_line"])

    print(str(disjoint_stats["log_line"]))
