#!/usr/bin/env python3
"""Asymptotic Monte Carlo experiment for adversarial structural estimation."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import stat
import shutil
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import degree, k_hop_subgraph, to_undirected


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ExperimentConfig, MonteCarloConfig
from constants import OPTIMAL_DISC_LOSS, OPTIMAL_GEN_LOSS
from discriminator import RootedMPNNDiscriminator
from generator import SCMGenerator
from io_utils import (
    append_realization_row,
    load_completed_realizations,
    save_json_manifest,
    save_realization_history,
)
from root_sampling import RootSampler, build_adjacency_from_edge_index, sample_roots_tensor
from utils import (
    build_row_stochastic_W,
    check_gan_convergence,
    compute_instance_noise_taus,
    extract_ego_batch,
)
from visualization import (
    plot_mc_parameter_distributions,
    plot_mc_quantile_convergence_paths,
    plot_mc_quantile_loss_paths,
)


DEVICE = torch.device("cpu")
DEBUG_MODE = os.environ.get("MC_DEBUG") == "1"


def build_graph_and_infrastructure(
    cfg: ExperimentConfig,
    mc_cfg: MonteCarloConfig,
) -> tuple[
    nx.Graph,
    Tensor,
    Tensor,
    Tensor,
    dict[int, tuple[Tensor, Tensor, int]],
    RootSampler,
    dict[str, float],
    int,
]:
    """Build fixed graph/covariate/sampling infrastructure shared across realizations.

    Args:
        cfg: Experiment configuration.
        mc_cfg: Monte Carlo configuration.

    Returns:
        Tuple containing graph, W, X, edge_index, ego_cache, root_sampler,
        normalization template, and node count.

    Raises:
        RuntimeError: If graph sanitization fails.
        ValueError: If graph type is incompatible with this experiment.
    """
    if cfg.graph.graph_type != "ba":
        raise ValueError(
            "Asymptotic Monte Carlo experiment expects graph_type='ba', got "
            f"{cfg.graph.graph_type!r}"
        )

    graph_seed = mc_cfg.master_seed
    covariate_seed = mc_cfg.master_seed + 1

    torch.manual_seed(graph_seed)
    np.random.seed(graph_seed)
    graph = nx.barabasi_albert_graph(
        n=cfg.graph.n_nodes,
        m=cfg.graph.ba_m,
        seed=graph_seed,
    )

    n_nodes_before_sanitize = graph.number_of_nodes()
    n_selfloops_removed = nx.number_of_selfloops(graph)
    if n_selfloops_removed > 0:
        graph.remove_edges_from(nx.selfloop_edges(graph))

    if not nx.is_connected(graph):
        gcc_nodes = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(gcc_nodes).copy()

    if set(graph.nodes()) != set(range(graph.number_of_nodes())):
        graph = nx.convert_node_labels_to_integers(
            graph,
            first_label=0,
            ordering="sorted",
        )

    isolates = [node for node, node_degree in graph.degree() if node_degree == 0]
    if isolates:
        raise RuntimeError(
            f"Graph has {len(isolates)} isolates after sanitization; cannot build W."
        )

    num_nodes = graph.number_of_nodes()
    edge_pairs = np.asarray(list(graph.edges()), dtype=np.int64)
    if edge_pairs.size == 0:
        raise RuntimeError("Graph has no edges after sanitization.")

    edge_index = torch.as_tensor(edge_pairs.T, dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes).contiguous().to(DEVICE)
    W = build_row_stochastic_W(edge_index=edge_index, num_nodes=num_nodes).to(DEVICE)

    degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
    if torch.any(degrees <= 0):
        raise RuntimeError("Graph contains non-positive degree nodes after sanitization.")
    if num_nodes > n_nodes_before_sanitize:
        raise RuntimeError("Sanitization increased node count unexpectedly.")

    torch.manual_seed(covariate_seed)
    X = torch.randn(num_nodes, device=DEVICE)

    ego_cache: dict[int, tuple[Tensor, Tensor, int]] = {}
    for root in range(num_nodes):
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx=root,
            num_hops=cfg.model.k,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        ego_cache[root] = (
            subset.to(DEVICE),
            sub_edge_index.to(DEVICE),
            int(mapping.item()),
        )

    sampler_seed = mc_cfg.master_seed + 2
    adjacency = build_adjacency_from_edge_index(edge_index=edge_index.cpu(), num_nodes=num_nodes)
    root_sampler = RootSampler(
        num_nodes=num_nodes,
        mode=cfg.training.resolved_root_sampler_mode(),
        exclusion_r=cfg.training.root_exclusion_r,
        disjoint_restarts_k=cfg.training.resolved_disjoint_restarts_k(),
        disjoint_min_batch=cfg.training.resolved_disjoint_min_batch(),
        disjoint_relax_sequence=cfg.training.resolved_disjoint_relax_sequence(),
        disjoint_fallback=cfg.training.disjoint_fallback,
        rng=np.random.default_rng(sampler_seed),
        adjacency=adjacency,
    )

    norm_stats_template = {
        "mu_X": float(X.mean().item()),
        "sigma_X": float(X.std(unbiased=False).item()),
    }
    return graph, W, X, edge_index, ego_cache, root_sampler, norm_stats_template, num_nodes


def _make_failure_result(
    realization_idx: int,
    step: int,
    history: dict[str, list[float]] | None,
    reason: str,
    *,
    gt_seed: int = -1,
    train_seed: int = -1,
) -> dict[str, Any]:
    """Create a standardized failure result payload.

    Args:
        realization_idx: Realization index.
        step: Step where failure occurred.
        history: Partial history if available.
        reason: Failure reason tag/message.
        gt_seed: Ground-truth seed used for this realization.
        train_seed: Training seed used for this realization.

    Returns:
        Failure payload with NaN estimates and retained history.
    """
    return {
        "realization": realization_idx,
        "converged": False,
        "final_step": int(step),
        "beta_hat": float("nan"),
        "gamma_hat": float("nan"),
        "sigma_sq_hat": float("nan"),
        "beta_final": float("nan"),
        "gamma_final": float("nan"),
        "sigma_sq_final": float("nan"),
        "loss_d_final": float("nan"),
        "loss_g_final": float("nan"),
        "loss_d_rolling_final": float("nan"),
        "loss_g_rolling_final": float("nan"),
        "init_seed": -1,
        "init_beta": float("nan"),
        "init_gamma": float("nan"),
        "init_log_sigma_sq": float("nan"),
        "gt_seed": int(gt_seed),
        "train_seed": int(train_seed),
        "status": f"failed:{reason}",
        "history": history if history is not None else {},
    }


def _derive_permuted_seed(master_seed: int, realization_idx: int, stream_id: int) -> int:
    """Derive deterministic per-realization seeds using independent seed streams."""
    seed_seq = np.random.SeedSequence([int(master_seed), int(realization_idx), int(stream_id)])
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def _sample_generator_initial_params(
    *,
    cfg: ExperimentConfig,
    mc_cfg: MonteCarloConfig,
    realization_idx: int,
) -> tuple[int, float, float, float]:
    """Sample per-realization generator initial values from uniform supports."""
    init_seed = _derive_permuted_seed(mc_cfg.master_seed, realization_idx, stream_id=3)

    rng = np.random.default_rng(init_seed)
    beta_low, beta_high = mc_cfg.init_uniform_beta_range
    gamma_low, gamma_high = mc_cfg.init_uniform_gamma_range
    log_sigma_low, log_sigma_high = mc_cfg.init_uniform_log_sigma_sq_range

    # Identification requires |beta| < 1 when W is row-stochastic (rho(W)=1).
    if float(beta_low) <= -1.0 or float(beta_high) >= 1.0:
        raise ValueError(
            "init_uniform_beta_range must satisfy -1 < low < high < 1 under "
            f"identification; got {mc_cfg.init_uniform_beta_range}."
        )
    # Generator reparameterization still enforces |beta| < beta_cap.
    if float(beta_low) <= -cfg.model.beta_cap or float(beta_high) >= cfg.model.beta_cap:
        raise ValueError(
            "init_uniform_beta_range must satisfy |beta| < beta_cap, got "
            f"{mc_cfg.init_uniform_beta_range} with beta_cap={cfg.model.beta_cap:.3f}."
        )

    init_beta = float(rng.uniform(float(beta_low), float(beta_high)))
    init_gamma = float(rng.uniform(float(gamma_low), float(gamma_high)))

    if mc_cfg.init_sigma_sq_fixed_unit:
        init_log_sigma_sq = 0.0
    else:
        init_log_sigma_sq = float(rng.uniform(float(log_sigma_low), float(log_sigma_high)))
    return init_seed, init_beta, init_gamma, init_log_sigma_sq


def _parameters_stabilized(
    history: dict[str, list[float]],
    *,
    window: int,
    beta_range_tol: float,
    gamma_range_tol: float,
    sigma_sq_range_tol: float,
) -> bool:
    """Check whether generator parameter paths are stable over a trailing window."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if len(history["beta"]) < window:
        return False

    beta_tail = history["beta"][-window:]
    gamma_tail = history["gamma"][-window:]
    sigma_tail = history["sigma_sq"][-window:]

    beta_range = max(beta_tail) - min(beta_tail)
    gamma_range = max(gamma_tail) - min(gamma_tail)
    sigma_range = max(sigma_tail) - min(sigma_tail)
    return (
        beta_range <= beta_range_tol
        and gamma_range <= gamma_range_tol
        and sigma_range <= sigma_sq_range_tol
    )


def _maybe_print_step_progress(
    *,
    realization_idx: int,
    step: int,
    theta: dict[str, float],
    loss_d: float,
    loss_g: float,
    history: dict[str, list[float]],
    mc_cfg: MonteCarloConfig,
) -> None:
    """Print per-step optimization diagnostics when configured."""
    every_n = mc_cfg.progress_every_n_steps
    if every_n is None or step % every_n != 0:
        return

    loss_window = min(mc_cfg.convergence_window, len(history["loss_d"]))
    if loss_window > 0:
        loss_d_roll = float(np.mean(history["loss_d"][-loss_window:]))
        loss_g_roll = float(np.mean(history["loss_g"][-loss_window:]))
    else:
        loss_d_roll = float("nan")
        loss_g_roll = float("nan")

    print(
        f"R{realization_idx:04d} step={step:04d} "
        f"beta={theta['beta']:.4f} gamma={theta['gamma']:.4f} "
        f"sigma_sq={theta['sigma_sq']:.4f} "
        f"loss=({loss_d:.4f}, {loss_g:.4f}) "
        f"loss_roll=({loss_d_roll:.4f}, {loss_g_roll:.4f})",
        flush=True,
    )


def run_single_realization(
    realization_idx: int,
    cfg: ExperimentConfig,
    mc_cfg: MonteCarloConfig,
    W: Tensor,
    X: Tensor,
    edge_index: Tensor,
    ego_cache: dict[int, tuple[Tensor, Tensor, int]],
    root_sampler: RootSampler,
    norm_stats_template: dict[str, float],
    num_nodes: int,
) -> dict[str, Any]:
    """Run one realization of ground-truth simulation + adversarial estimation.

    Args:
        realization_idx: Realization index in ``[0, n_realizations)``.
        cfg: Experiment configuration.
        mc_cfg: Monte Carlo configuration.
        W: Row-stochastic sparse matrix.
        X: Fixed covariate vector.
        edge_index: Graph edge index.
        ego_cache: Precomputed rooted ego-graph cache.
        root_sampler: Root sampler (RNG is reseeded in this function).
        norm_stats_template: Fixed ``mu_X``/``sigma_X`` values.
        num_nodes: Graph node count.

    Returns:
        Realization result dictionary with per-step history.
    """
    del edge_index  # Included for signature clarity; graph topology is encoded in W/ego_cache.
    del num_nodes

    gt_seed = _derive_permuted_seed(mc_cfg.master_seed, realization_idx, stream_id=0)
    train_seed = _derive_permuted_seed(mc_cfg.master_seed, realization_idx, stream_id=1)
    sampler_seed = _derive_permuted_seed(mc_cfg.master_seed, realization_idx, stream_id=2)
    root_sampler.rng = np.random.default_rng(sampler_seed)

    history: dict[str, list[float]] = {
        "beta": [],
        "gamma": [],
        "sigma_sq": [],
        "loss_d": [],
        "loss_g": [],
        "tau_x": [],
        "tau_y": [],
    }

    generator: SCMGenerator | None = None
    discriminator: RootedMPNNDiscriminator | None = None
    opt_d: torch.optim.Adam | None = None
    opt_g: torch.optim.Adam | None = None
    Y_obs: Tensor | None = None
    true_generator: SCMGenerator | None = None

    try:
        torch.manual_seed(gt_seed)
        true_generator = SCMGenerator(
            beta_cap=cfg.model.beta_cap,
            picard_tol=cfg.model.picard_tol,
            picard_max=cfg.model.picard_max,
            init_beta=cfg.true_params.beta,
            init_gamma=cfg.true_params.gamma,
            init_log_sigma_sq=math.log(cfg.true_params.sigma_sq),
        ).to(DEVICE)
        with torch.no_grad():
            Y_obs = true_generator(W, X)
        assert torch.isfinite(Y_obs).all(), (
            f"Realization {realization_idx}: Y_obs contains non-finite values"
        )

        norm_stats = {
            **norm_stats_template,
            "mu_Y": float(Y_obs.mean().item()),
            "sigma_Y": float(Y_obs.std(unbiased=False).item()),
        }
        assert norm_stats["sigma_Y"] > 1e-10, (
            f"Realization {realization_idx}: sigma_Y is near-zero"
        )

        init_seed, init_beta, init_gamma, init_log_sigma_sq = _sample_generator_initial_params(
            cfg=cfg,
            mc_cfg=mc_cfg,
            realization_idx=realization_idx,
        )

        torch.manual_seed(train_seed)
        np.random.seed(train_seed)

        generator = SCMGenerator(
            beta_cap=cfg.model.beta_cap,
            picard_tol=cfg.model.picard_tol,
            picard_max=cfg.model.picard_max,
            init_beta=init_beta,
            init_gamma=init_gamma,
            init_log_sigma_sq=init_log_sigma_sq,
        ).to(DEVICE)
        discriminator = RootedMPNNDiscriminator(
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.k,
            logit_clip=cfg.model.logit_clip,
        ).to(DEVICE)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.training.lr_d)
        opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.training.lr_g)

        converged = False
        final_step = int(mc_cfg.max_steps) if mc_cfg.max_steps is not None else 0
        is_in_equilibrium = False

        step = 0
        while True:
            step += 1
            if step in mc_cfg.lr_g_decay_steps:
                for group in opt_g.param_groups:
                    group["lr"] *= mc_cfg.lr_g_decay_factor

            active_instance_noise = cfg.instance_noise
            if (
                cfg.instance_noise.enabled
                and mc_cfg.adaptive_anneal_enabled
                and cfg.instance_noise.anneal_steps > 0
                and step > cfg.instance_noise.anneal_steps
                and not is_in_equilibrium
            ):
                adaptive_anneal_steps = step + mc_cfg.adaptive_anneal_buffer_steps
                active_instance_noise = replace(
                    cfg.instance_noise,
                    anneal_steps=adaptive_anneal_steps,
                )

            discriminator.train()
            for param in discriminator.parameters():
                param.requires_grad_(True)

            with torch.no_grad():
                Y_sim_detached = generator(W, X)
            if not torch.isfinite(Y_sim_detached).all():
                warnings.warn(
                    f"R{realization_idx} step {step}: Y_sim non-finite in D phase.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return _make_failure_result(
                    realization_idx,
                    step,
                    history,
                    "Y_sim_non_finite_D",
                    gt_seed=gt_seed,
                    train_seed=train_seed,
                )

            last_loss_d = float("nan")
            for _ in range(cfg.training.n_disc):
                roots, _ = sample_roots_tensor(
                    sampler=root_sampler,
                    batch_size=cfg.training.batch_size,
                    device=DEVICE,
                )
                batch_obs, root_idx_obs = extract_ego_batch(
                    roots=roots,
                    ego_cache=ego_cache,
                    X=X,
                    Y=Y_obs,
                    norm_stats=norm_stats,
                    instance_noise=active_instance_noise,
                    generator_step=step,
                    batch_role="real",
                )
                batch_sim, root_idx_sim = extract_ego_batch(
                    roots=roots,
                    ego_cache=ego_cache,
                    X=X,
                    Y=Y_sim_detached,
                    norm_stats=norm_stats,
                    instance_noise=active_instance_noise,
                    generator_step=step,
                    batch_role="fake",
                )

                opt_d.zero_grad(set_to_none=True)
                logits_real = discriminator(batch_obs.x, batch_obs.edge_index, root_idx_obs)
                logits_fake = discriminator(batch_sim.x, batch_sim.edge_index, root_idx_sim)
                loss_d = F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean()
                loss_d.backward()
                opt_d.step()
                last_loss_d = float(loss_d.item())

            for param in discriminator.parameters():
                param.requires_grad_(False)

            roots_g, _ = sample_roots_tensor(
                sampler=root_sampler,
                batch_size=cfg.training.batch_size,
                device=DEVICE,
            )
            Y_sim = generator(W, X)
            if not torch.isfinite(Y_sim).all():
                warnings.warn(
                    f"R{realization_idx} step {step}: Y_sim non-finite in G phase.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return _make_failure_result(
                    realization_idx,
                    step,
                    history,
                    "Y_sim_non_finite_G",
                    gt_seed=gt_seed,
                    train_seed=train_seed,
                )

            batch_sim_g, root_idx_sim_g = extract_ego_batch(
                roots=roots_g,
                ego_cache=ego_cache,
                X=X,
                Y=Y_sim,
                norm_stats=norm_stats,
                instance_noise=active_instance_noise,
                generator_step=step,
                batch_role="fake",
            )
            opt_g.zero_grad(set_to_none=True)
            logits_fake_g = discriminator(batch_sim_g.x, batch_sim_g.edge_index, root_idx_sim_g)
            loss_g = F.softplus(-logits_fake_g).mean()
            loss_g.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=mc_cfg.grad_clip_norm)
            opt_g.step()

            theta = generator.get_params()
            tau_x, tau_y = compute_instance_noise_taus(
                instance_noise=active_instance_noise,
                generator_step=step,
            )
            loss_g_value = float(loss_g.item())

            history["beta"].append(theta["beta"])
            history["gamma"].append(theta["gamma"])
            history["sigma_sq"].append(theta["sigma_sq"])
            history["loss_d"].append(last_loss_d)
            history["loss_g"].append(loss_g_value)
            history["tau_x"].append(float(tau_x))
            history["tau_y"].append(float(tau_y))

            if math.isnan(last_loss_d) or math.isnan(loss_g_value):
                warnings.warn(
                    f"R{realization_idx} step {step}: NaN loss detected.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return _make_failure_result(
                    realization_idx,
                    step,
                    history,
                    "nan_loss",
                    gt_seed=gt_seed,
                    train_seed=train_seed,
                )

            _maybe_print_step_progress(
                realization_idx=realization_idx,
                step=step,
                theta=theta,
                loss_d=last_loss_d,
                loss_g=loss_g_value,
                history=history,
                mc_cfg=mc_cfg,
            )

            if DEBUG_MODE and step % 50 == 0:
                assert abs(theta["beta"]) < cfg.model.beta_cap, "beta escaped beta_cap"
                assert theta["sigma_sq"] > 0.0, "sigma_sq became non-positive"
                assert all(math.isfinite(value) for value in theta.values()), (
                    "Generator parameters became non-finite"
                )

            is_in_equilibrium, _, _ = check_gan_convergence(
                loss_d_history=history["loss_d"],
                loss_g_history=history["loss_g"],
                window=mc_cfg.convergence_window,
                delta_d=mc_cfg.convergence_delta_d,
                delta_g=mc_cfg.convergence_delta_g,
                min_steps=mc_cfg.min_steps,
                std_d_max=mc_cfg.convergence_std_d_max,
                std_g_max=mc_cfg.convergence_std_g_max,
            )

            has_stable_params = _parameters_stabilized(
                history=history,
                window=mc_cfg.stability_window,
                beta_range_tol=mc_cfg.stability_beta_range_tol,
                gamma_range_tol=mc_cfg.stability_gamma_range_tol,
                sigma_sq_range_tol=mc_cfg.stability_sigma_sq_range_tol,
            )
            if is_in_equilibrium and has_stable_params:
                converged = True
                final_step = step
                break
            if mc_cfg.max_steps is not None and step >= mc_cfg.max_steps:
                final_step = step
                break

        tail_window = min(
            max(mc_cfg.convergence_window, mc_cfg.stability_window),
            len(history["beta"]),
        )
        beta_hat = float(np.mean(history["beta"][-tail_window:]))
        gamma_hat = float(np.mean(history["gamma"][-tail_window:]))
        sigma_sq_hat = float(np.mean(history["sigma_sq"][-tail_window:]))
        if abs(beta_hat) >= cfg.model.beta_cap:
            warnings.warn(
                f"R{realization_idx}: |beta_hat| exceeded beta_cap ({beta_hat:.4f}).",
                RuntimeWarning,
                stacklevel=2,
            )

        loss_window = min(mc_cfg.convergence_window, len(history["loss_d"]))
        if loss_window > 0:
            loss_d_rolling_final = float(np.mean(history["loss_d"][-loss_window:]))
            loss_g_rolling_final = float(np.mean(history["loss_g"][-loss_window:]))
        else:
            loss_d_rolling_final = float("nan")
            loss_g_rolling_final = float("nan")

        return {
            "realization": realization_idx,
            "converged": converged,
            "final_step": final_step,
            "beta_hat": beta_hat,
            "gamma_hat": gamma_hat,
            "sigma_sq_hat": sigma_sq_hat,
            "beta_final": history["beta"][-1],
            "gamma_final": history["gamma"][-1],
            "sigma_sq_final": history["sigma_sq"][-1],
            "loss_d_final": history["loss_d"][-1],
            "loss_g_final": history["loss_g"][-1],
            "loss_d_rolling_final": loss_d_rolling_final,
            "loss_g_rolling_final": loss_g_rolling_final,
            "init_seed": init_seed,
            "init_beta": init_beta,
            "init_gamma": init_gamma,
            "init_log_sigma_sq": init_log_sigma_sq,
            "gt_seed": gt_seed,
            "train_seed": train_seed,
            "status": "ok",
            "history": history,
        }
    except Exception as exc:  # pragma: no cover - defensive catch for long runs.
        warnings.warn(
            f"Realization {realization_idx} failed with exception: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _make_failure_result(
            realization_idx,
            len(history["beta"]),
            history,
            str(exc),
            gt_seed=gt_seed,
            train_seed=train_seed,
        )
    finally:
        if true_generator is not None:
            del true_generator
        if generator is not None:
            del generator
        if discriminator is not None:
            del discriminator
        if opt_d is not None:
            del opt_d
        if opt_g is not None:
            del opt_g
        if Y_obs is not None:
            del Y_obs
        gc.collect()


def load_all_histories(
    hist_dir: Path,
    results: list[dict[str, Any]],
) -> list[dict[str, np.ndarray]]:
    """Load and edge-pad histories for quantile computations.

    Args:
        hist_dir: Directory containing ``history_rXXXX.npz`` files.
        results: Completed result rows.

    Returns:
        List of padded history dictionaries.
    """
    ok_results = [row for row in results if row.get("status") == "ok"]
    if not ok_results:
        return []

    max_len = max(int(row["final_step"]) for row in ok_results)
    if max_len <= 0:
        return []

    keys = ("beta", "gamma", "sigma_sq", "loss_d", "loss_g")
    histories: list[dict[str, np.ndarray]] = []
    for row in ok_results:
        realization = int(row["realization"])
        path = hist_dir / f"history_r{realization:04d}.npz"
        if not path.exists():
            warnings.warn(
                f"Missing history file for realization {realization}: {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        with np.load(path) as data:
            padded: dict[str, np.ndarray] = {}
            valid = True
            for key in keys:
                if key not in data:
                    valid = False
                    break
                arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
                if arr.size == 0:
                    valid = False
                    break
                if arr.size < max_len:
                    arr = np.pad(arr, (0, max_len - arr.size), mode="edge")
                else:
                    arr = arr[:max_len]
                padded[key] = arr
            if valid:
                histories.append(padded)
    return histories


def _resolve_mc_config(mc_cfg: MonteCarloConfig) -> MonteCarloConfig:
    """Resolve CLI/env overrides for Monte Carlo run controls.

    Args:
        mc_cfg: Baseline Monte Carlo configuration.

    Returns:
        Monte Carlo configuration with optional overrides applied.

    Raises:
        SystemExit: If command-line arguments or environment overrides are invalid.
    """
    resolved = mc_cfg
    if len(sys.argv) not in (1, 2):
        raise SystemExit(
            "Usage: asymptotic_mc_experiment.py [n_realizations_override]"
        )
    if len(sys.argv) == 2:
        try:
            n_realizations_override = int(sys.argv[1])
        except ValueError as exc:
            raise SystemExit(
                f"n_realizations_override must be an integer, got {sys.argv[1]!r}"
            ) from exc
        resolved = replace(resolved, n_realizations=n_realizations_override)

    progress_override = os.environ.get("MC_PROGRESS_EVERY_STEPS")
    if progress_override is None:
        return resolved
    try:
        progress_steps = int(progress_override)
    except ValueError as exc:
        raise SystemExit(
            "MC_PROGRESS_EVERY_STEPS must be an integer (0 disables progress logs), "
            f"got {progress_override!r}"
        ) from exc
    if progress_steps < 0:
        raise SystemExit(
            "MC_PROGRESS_EVERY_STEPS must be >= 0 (0 disables progress logs), "
            f"got {progress_steps}"
        )
    if progress_steps == 0:
        return replace(resolved, progress_every_n_steps=None)
    return replace(resolved, progress_every_n_steps=progress_steps)


def _summarize_parameter_estimates(
    ok_results: list[dict[str, Any]],
    true_params: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for parameter estimates across realizations."""
    summary: dict[str, dict[str, float]] = {}
    for key in ("beta", "gamma", "sigma_sq"):
        est_key = f"{key}_hat"
        values = np.asarray(
            [float(row[est_key]) for row in ok_results if math.isfinite(float(row[est_key]))],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        errors = values - float(true_params[key])
        summary[key] = {
            "true": float(true_params[key]),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "bias": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "median": float(np.median(values)),
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        }
    return summary


def _load_saved_run_config(path: Path) -> dict[str, Any] | None:
    """Load previously saved Monte Carlo config snapshot if present.

    Args:
        path: Path to ``mc_config.json``.

    Returns:
        Parsed config dictionary, or ``None`` if the file is missing/corrupt.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _remove_path_with_permission_fix(path: Path) -> None:
    """Remove a file/directory, retrying once after chmod on permission errors.

    Args:
        path: Path to remove.

    Raises:
        OSError: If removal fails after permission-fix retry.
    """
    if not path.exists():
        return

    def _on_rm_error(func: Any, failing_path: str, _: Any) -> None:
        os.chmod(failing_path, stat.S_IWRITE)
        func(failing_path)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onerror=_on_rm_error)
    else:
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()


def _clear_mc_output_artifacts(output_dir: Path) -> None:
    """Clear run artifacts that can cause cross-parameterization mixing.

    This intentionally avoids deleting the entire output root because OneDrive/
    Windows file locks can make full directory removal flaky. A clean run is
    still guaranteed by removing critical resume/config files and stale histories.

    Args:
        output_dir: Monte Carlo output directory.

    Raises:
        RuntimeError: If critical files cannot be removed.
    """
    critical_files = (
        output_dir / "mc_results.csv",
        output_dir / "mc_manifest.json",
        output_dir / "mc_config.json",
    )
    for file_path in critical_files:
        if not file_path.exists():
            continue
        try:
            _remove_path_with_permission_fix(file_path)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot remove critical file {file_path}. "
                "Another run may still be active."
            ) from exc

    for figure_path in output_dir.glob("fig_*.png"):
        try:
            _remove_path_with_permission_fix(figure_path)
        except OSError:
            warnings.warn(
                f"Could not remove stale figure {figure_path}; continuing.",
                RuntimeWarning,
                stacklevel=2,
            )

    histories_dir = output_dir / "histories"
    if histories_dir.exists():
        for history_file in histories_dir.glob("history_r*.npz"):
            try:
                _remove_path_with_permission_fix(history_file)
            except OSError:
                warnings.warn(
                    f"Could not remove stale history {history_file}; continuing.",
                    RuntimeWarning,
                    stacklevel=2,
                )


def _purge_output_dir_if_config_changed(
    output_dir: Path,
    desired_config: dict[str, Any],
) -> bool:
    """Purge existing Monte Carlo artifacts when config differs from prior runs.

    Args:
        output_dir: Root output directory for Monte Carlo artifacts.
        desired_config: Effective config snapshot for this run.

    Returns:
        ``True`` if existing artifacts were purged, else ``False``.
    """
    def _normalize_for_compare(config: dict[str, Any]) -> dict[str, Any]:
        """Normalize run config for purge comparisons.

        `n_realizations` and plot cadence are run-budget controls; changing them
        alone should not invalidate existing Monte Carlo artifacts.
        """
        normalized = deepcopy(config)
        mc_section = normalized.get("monte_carlo")
        if isinstance(mc_section, dict):
            mc_section.pop("n_realizations", None)
            mc_section.pop("plot_every_n_realizations", None)
            mc_section.pop("progress_every_n_steps", None)
        # Canonicalize Python-only container types (e.g., tuple -> list) to match
        # how configs are persisted to JSON on disk.
        return json.loads(json.dumps(normalized))

    if not output_dir.exists():
        return False

    config_path = output_dir / "mc_config.json"
    has_any_artifacts = any(output_dir.iterdir())
    if not has_any_artifacts:
        return False

    saved_config = _load_saved_run_config(config_path)
    should_purge = False
    reason = ""

    if saved_config is None:
        should_purge = True
        reason = "missing_or_corrupt_mc_config"
    elif _normalize_for_compare(saved_config) != _normalize_for_compare(desired_config):
        should_purge = True
        reason = "config_changed"

    if not should_purge:
        return False

    print(
        "Detected Monte Carlo configuration mismatch; purging previous artifacts "
        f"from {output_dir} (reason={reason})."
    )
    _clear_mc_output_artifacts(output_dir)
    return True


def _generate_mc_visualizations(
    *,
    output_dir: Path,
    histories_dir: Path,
    results: list[dict[str, Any]],
    true_params: dict[str, float],
    max_steps: int | None,
) -> None:
    """Generate/rewrite Monte Carlo charts from currently available rows."""
    ok_results = [row for row in results if row.get("status") == "ok"]
    if len(ok_results) < 2:
        return
    if max_steps is None:
        plot_horizon = max(int(row.get("final_step", 0)) for row in ok_results)
    else:
        plot_horizon = int(max_steps)
    if plot_horizon <= 0:
        return

    plot_mc_parameter_distributions(
        results=ok_results,
        true_params=true_params,
        save_path=output_dir / "fig_param_distributions.png",
    )
    histories = load_all_histories(hist_dir=histories_dir, results=ok_results)
    if not histories:
        return
    plot_mc_quantile_convergence_paths(
        histories=histories,
        true_params=true_params,
        max_steps=plot_horizon,
        save_path=output_dir / "fig_param_convergence_quantiles.png",
    )
    plot_mc_quantile_loss_paths(
        histories=histories,
        max_steps=plot_horizon,
        save_path=output_dir / "fig_loss_convergence_quantiles.png",
    )


def main() -> None:
    """Run the asymptotic Monte Carlo experiment."""
    cfg = ExperimentConfig.mc_default()
    mc_cfg = _resolve_mc_config(MonteCarloConfig())

    output_dir = REPO_ROOT / mc_cfg.output_dir
    desired_run_config = {
        "experiment": cfg.to_dict(),
        "monte_carlo": asdict(mc_cfg),
    }
    _purge_output_dir_if_config_changed(
        output_dir=output_dir,
        desired_config=desired_run_config,
    )

    histories_dir = output_dir / "histories"
    output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "mc_results.csv"
    config_path = output_dir / "mc_config.json"
    manifest_path = output_dir / "mc_manifest.json"

    save_json_manifest(
        config_path,
        desired_run_config,
    )

    print("Building graph and shared infrastructure...")
    (
        _graph,
        W,
        X,
        edge_index,
        ego_cache,
        root_sampler,
        norm_stats_template,
        num_nodes,
    ) = build_graph_and_infrastructure(cfg=cfg, mc_cfg=mc_cfg)

    assert len(ego_cache) == num_nodes, "ego_cache incomplete"
    assert torch.isfinite(X).all(), "X contains non-finite values"
    assert W.shape == (num_nodes, num_nodes) and W.is_sparse, "W shape/layout mismatch"
    assert norm_stats_template["sigma_X"] > 1e-10, "sigma_X is non-positive"
    print(
        f"Pre-flight checks passed. nodes={num_nodes}, edges={edge_index.shape[1] // 2}, "
        f"ego_cache={len(ego_cache)}"
    )
    if mc_cfg.progress_every_n_steps is not None:
        print(
            "Per-step progress logging enabled: "
            f"every {mc_cfg.progress_every_n_steps} generator steps."
        )

    completed_rows = load_completed_realizations(results_path)
    completed: set[int] = {
        int(row["realization"]) for row in completed_rows if "realization" in row
    }
    if completed:
        print(f"Resuming run: {len(completed)} realizations already logged.")

    true_params = {
        "beta": cfg.true_params.beta,
        "gamma": cfg.true_params.gamma,
        "sigma_sq": cfg.true_params.sigma_sq,
    }
    total_logged_rows = len(completed_rows)

    run_start = time.time()
    for realization_idx in range(mc_cfg.n_realizations):
        if realization_idx in completed:
            continue

        print(f"\n{'=' * 60}")
        print(f"Realization {realization_idx}/{mc_cfg.n_realizations - 1}")
        print(f"{'=' * 60}")

        t0 = time.time()
        result = run_single_realization(
            realization_idx=realization_idx,
            cfg=cfg,
            mc_cfg=mc_cfg,
            W=W,
            X=X,
            edge_index=edge_index,
            ego_cache=ego_cache,
            root_sampler=root_sampler,
            norm_stats_template=norm_stats_template,
            num_nodes=num_nodes,
        )
        elapsed = time.time() - t0

        history = result.pop("history", None)
        if isinstance(history, dict) and any(len(values) > 0 for values in history.values()):
            history_path = histories_dir / f"history_r{realization_idx:04d}.npz"
            save_realization_history(history_path, history)
        result["elapsed_seconds"] = round(elapsed, 2)
        append_realization_row(results_path, result)
        total_logged_rows += 1

        beta_hat = float(result.get("beta_hat", float("nan")))
        gamma_hat = float(result.get("gamma_hat", float("nan")))
        sigma_sq_hat = float(result.get("sigma_sq_hat", float("nan")))
        loss_d_rolling = float(result.get("loss_d_rolling_final", float("nan")))
        loss_g_rolling = float(result.get("loss_g_rolling_final", float("nan")))
        print(
            f"beta_hat={beta_hat:.4f}  gamma_hat={gamma_hat:.4f}  "
            f"sigma_sq_hat={sigma_sq_hat:.4f}  "
            f"steps={int(result.get('final_step', -1))}  "
            f"converged={bool(result.get('converged', False))}  "
            f"loss_roll=({loss_d_rolling:.4f}, {loss_g_rolling:.4f})  "
            f"time={elapsed:.1f}s  status={result.get('status', 'unknown')}"
        )

        if total_logged_rows % mc_cfg.plot_every_n_realizations == 0:
            print(
                "Refreshing Monte Carlo visualizations at "
                f"{total_logged_rows} logged realizations..."
            )
            partial_results = load_completed_realizations(results_path)
            _generate_mc_visualizations(
                output_dir=output_dir,
                histories_dir=histories_dir,
                results=partial_results,
                true_params=true_params,
                max_steps=mc_cfg.max_steps,
            )
        gc.collect()

    print("\nGenerating Monte Carlo visualizations...")
    all_results = load_completed_realizations(results_path)
    _generate_mc_visualizations(
        output_dir=output_dir,
        histories_dir=histories_dir,
        results=all_results,
        true_params=true_params,
        max_steps=mc_cfg.max_steps,
    )
    ok_results = [row for row in all_results if row.get("status") == "ok"]

    converged_steps = [
        int(row["final_step"])
        for row in ok_results
        if bool(row.get("converged", False))
    ]
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - run_start, 2),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(DEVICE),
        "n_requested": mc_cfg.n_realizations,
        "n_total_rows": len(all_results),
        "n_completed": len(ok_results),
        "n_failed": len(all_results) - len(ok_results),
        "n_converged": len(converged_steps),
        "median_convergence_step": (
            float(np.median(converged_steps)) if converged_steps else float("nan")
        ),
        "parameter_summary": _summarize_parameter_estimates(ok_results, true_params),
        "equilibrium_targets": {
            "disc_loss": OPTIMAL_DISC_LOSS,
            "gen_loss": OPTIMAL_GEN_LOSS,
        },
    }
    save_json_manifest(manifest_path, manifest)

    print(
        f"\nDone. successful={len(ok_results)} failed={len(all_results) - len(ok_results)} "
        f"converged={len(converged_steps)}"
    )


if __name__ == "__main__":
    main()
