#!/usr/bin/env python3
"""Asymptotic Monte Carlo experiment for adversarial structural estimation.

Thin client over the estimation engine. All orchestration (seed streams, resume,
durable logging, provenance, periodic figures) lives in
:class:`src.runner.MonteCarloRunner`; this script only supplies the linear-in-
means specifics — building the shared substrate, simulating the observed outcome
from the true model, and constructing the fresh model + discriminator + config
for each realisation — plus the visualisation hook.

Usage:
    python asymptotic_mc_experiment.py [n_realizations_override]

Environment:
    MC_SMOKE=1               run a small, fast configuration (for verification)
    MC_PROGRESS_EVERY_STEPS  print per-step progress every N generator steps
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adversarial_networks import (  # noqa: E402  (path bootstrap must precede import)
    ConsoleLogger,
    EstimatorConfig,
    LinearInMeansGenerator,
    RootedMPNNDiscriminator,
)
from adversarial_networks.config import ExperimentConfig, MonteCarloConfig  # noqa: E402
from adversarial_networks.ego import EgoSubstrate  # noqa: E402
from adversarial_networks.provenance import config_hash  # noqa: E402
from adversarial_networks.runner import MonteCarloRunner, RealizationResult, RealizationSpec  # noqa: E402
from adversarial_networks.visualization import (  # noqa: E402
    plot_mc_parameter_distributions,
    plot_mc_quantile_convergence_paths,
    plot_mc_quantile_loss_paths,
)

DEVICE = torch.device("cpu")
_HISTORY_KEYS = ("beta", "gamma", "sigma_sq", "loss_d", "loss_g")


def build_substrate(cfg: ExperimentConfig, mc_cfg: MonteCarloConfig) -> EgoSubstrate:
    """Build the shared graph + covariate substrate (once, reused across realisations)."""
    if cfg.graph.graph_type != "ba":
        raise ValueError(f"Expected graph_type='ba', got {cfg.graph.graph_type!r}.")
    graph = nx.barabasi_albert_graph(n=cfg.graph.n_nodes, m=cfg.graph.ba_m, seed=mc_cfg.master_seed)
    torch.manual_seed(mc_cfg.master_seed + 1)
    X = torch.randn(graph.number_of_nodes(), device=DEVICE)
    return EgoSubstrate.from_networkx(
        graph,
        X,
        k=cfg.model.k,
        root_sampler_mode=cfg.training.resolved_root_sampler_mode(),
        exclusion_r=cfg.training.root_exclusion_r,
        disjoint_restarts_k=cfg.training.resolved_disjoint_restarts_k(),
        disjoint_min_batch=cfg.training.resolved_disjoint_min_batch(),
        disjoint_relax_sequence=cfg.training.resolved_disjoint_relax_sequence(),
        disjoint_fallback=cfg.training.disjoint_fallback,
        seed=mc_cfg.master_seed + 2,
    )


def make_observed_factory(cfg: ExperimentConfig):
    """Return a factory that simulates the observed outcome from the true model."""

    def observed(substrate: EgoSubstrate, gt_seed: int) -> torch.Tensor:
        torch.manual_seed(gt_seed)
        true_model = LinearInMeansGenerator(
            beta_cap=cfg.model.beta_cap,
            picard_tol=cfg.model.picard_tol,
            picard_max=cfg.model.picard_max,
            init_beta=cfg.true_params.beta,
            init_gamma=cfg.true_params.gamma,
            init_log_sigma_sq=math.log(cfg.true_params.sigma_sq),
        ).to(DEVICE)
        with torch.no_grad():
            return true_model(substrate.W, substrate.X)

    return observed


def _sample_initial_params(cfg: ExperimentConfig, mc_cfg: MonteCarloConfig, init_seed: int) -> dict[str, float]:
    """Sample per-realisation initial generator parameters from the configured supports."""
    rng = np.random.default_rng(init_seed)
    beta_low, beta_high = mc_cfg.init_uniform_beta_range
    gamma_low, gamma_high = mc_cfg.init_uniform_gamma_range
    log_s_low, log_s_high = mc_cfg.init_uniform_log_sigma_sq_range
    if beta_low <= -cfg.model.beta_cap or beta_high >= cfg.model.beta_cap:
        raise ValueError(
            f"init_uniform_beta_range must satisfy |beta| < beta_cap={cfg.model.beta_cap}."
        )
    init_beta = float(rng.uniform(float(beta_low), float(beta_high)))
    init_gamma = float(rng.uniform(float(gamma_low), float(gamma_high)))
    init_log_sigma_sq = 0.0 if mc_cfg.init_sigma_sq_fixed_unit else float(rng.uniform(float(log_s_low), float(log_s_high)))
    return {"beta": init_beta, "gamma": init_gamma, "log_sigma_sq": init_log_sigma_sq}


def make_estimator_factory(cfg: ExperimentConfig, mc_cfg: MonteCarloConfig):
    """Return a factory building the fresh model + discriminator + config per realisation."""
    base_config = EstimatorConfig.from_configs(cfg, mc_cfg)
    progress_every = mc_cfg.progress_every_n_steps

    def factory(substrate: EgoSubstrate, Y_obs: torch.Tensor, idx: int, seeds: Mapping[str, int]) -> RealizationSpec:
        init = _sample_initial_params(cfg, mc_cfg, seeds["init"])
        model = LinearInMeansGenerator(
            beta_cap=cfg.model.beta_cap,
            picard_tol=cfg.model.picard_tol,
            picard_max=cfg.model.picard_max,
            init_beta=init["beta"],
            init_gamma=init["gamma"],
            init_log_sigma_sq=init["log_sigma_sq"],
        ).to(DEVICE)
        discriminator = RootedMPNNDiscriminator(
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.k,
            logit_clip=cfg.model.logit_clip,
        ).to(DEVICE)
        config = replace(base_config, seed=int(seeds["train"]))
        observers = (ConsoleLogger(every_n_steps=progress_every, prefix=f"R{idx:04d}"),) if progress_every else ()
        return RealizationSpec(
            model=model,
            discriminator=discriminator,
            config=config,
            init_params=init,
            instance_noise=cfg.instance_noise,
            observers=observers,
        )

    return factory


def _load_histories(hist_dir: Path, results: list[RealizationResult]) -> list[dict[str, np.ndarray]]:
    """Load and edge-pad per-realisation histories for the quantile plots."""
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return []
    max_len = max(int(r.result.final_step) for r in ok)
    if max_len <= 0:
        return []
    histories: list[dict[str, np.ndarray]] = []
    for r in ok:
        path = hist_dir / f"history_r{r.realization:04d}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            if not all(key in data for key in _HISTORY_KEYS):
                continue
            padded: dict[str, np.ndarray] = {}
            for key in _HISTORY_KEYS:
                arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
                if arr.size == 0:
                    padded = {}
                    break
                arr = np.pad(arr, (0, max_len - arr.size), mode="edge") if arr.size < max_len else arr[:max_len]
                padded[key] = arr
            if padded:
                histories.append(padded)
    return histories


def make_visualize_hook(true_params: dict[str, float], max_steps: int | None):
    """Return the runner's visualisation hook reusing the project plot functions."""

    def visualize(output_dir: Path, results: list[RealizationResult]) -> None:
        ok = [r for r in results if r.status == "ok"]
        if len(ok) < 2:
            return
        rows = [r.to_row() for r in ok]
        plot_mc_parameter_distributions(
            results=rows, true_params=true_params, save_path=output_dir / "fig_param_distributions.png"
        )
        histories = _load_histories(output_dir / "histories", ok)
        if not histories:
            return
        horizon = int(max_steps) if max_steps is not None else max(int(r.result.final_step) for r in ok)
        plot_mc_quantile_convergence_paths(
            histories=histories, true_params=true_params, max_steps=horizon,
            save_path=output_dir / "fig_param_convergence_quantiles.png",
        )
        plot_mc_quantile_loss_paths(
            histories=histories, max_steps=horizon,
            save_path=output_dir / "fig_loss_convergence_quantiles.png",
        )

    return visualize


def _apply_overrides(cfg: ExperimentConfig, mc_cfg: MonteCarloConfig) -> tuple[ExperimentConfig, MonteCarloConfig]:
    """Apply CLI (n_realizations) and environment (MC_SMOKE / progress) overrides."""
    if len(sys.argv) not in (1, 2):
        raise SystemExit("Usage: asymptotic_mc_experiment.py [n_realizations_override]")
    if len(sys.argv) == 2:
        try:
            mc_cfg = replace(mc_cfg, n_realizations=int(sys.argv[1]))
        except ValueError as exc:
            raise SystemExit(f"n_realizations_override must be an integer, got {sys.argv[1]!r}") from exc

    progress_override = os.environ.get("MC_PROGRESS_EVERY_STEPS")
    if progress_override is not None:
        try:
            steps = int(progress_override)
        except ValueError as exc:
            raise SystemExit("MC_PROGRESS_EVERY_STEPS must be an integer.") from exc
        mc_cfg = replace(mc_cfg, progress_every_n_steps=(steps if steps > 0 else None))

    if os.environ.get("MC_SMOKE") == "1":
        cfg = replace(cfg, graph=replace(cfg.graph, n_nodes=300))
        # Keep MonteCarloConfig internally consistent: earliest feasible stop is
        # max(convergence_window + dwell - 1, stability_window, min_steps + dwell - 1).
        mc_cfg = replace(
            mc_cfg, n_realizations=min(mc_cfg.n_realizations, 3), max_steps=40, min_steps=0,
            convergence_window=20, stability_window=10, equilibrium_dwell_steps=5,
            plot_every_n_realizations=2, lr_g_decay_steps=(20,),
        )
    return cfg, mc_cfg


def main() -> None:
    """Run the asymptotic Monte Carlo experiment through the engine runner."""
    cfg = ExperimentConfig.mc_default()
    mc_cfg = MonteCarloConfig()
    cfg, mc_cfg = _apply_overrides(cfg, mc_cfg)

    run_config: dict[str, Any] = {"experiment": cfg.to_dict(), "monte_carlo": asdict(mc_cfg)}
    output_dir = REPO_ROOT / mc_cfg.output_dir
    _purge_if_config_changed(output_dir, run_config)

    print("Building shared graph and substrate...")
    substrate = build_substrate(cfg, mc_cfg)
    print(f"Substrate ready: nodes={substrate.num_nodes}, k={substrate.k}, sampler={substrate.root_sampler.mode!r}")

    true_params = {"beta": cfg.true_params.beta, "gamma": cfg.true_params.gamma, "sigma_sq": cfg.true_params.sigma_sq}
    runner = MonteCarloRunner(
        substrate=substrate,
        n_realizations=mc_cfg.n_realizations,
        master_seed=mc_cfg.master_seed,
        observed_factory=make_observed_factory(cfg),
        estimator_factory=make_estimator_factory(cfg, mc_cfg),
        output_dir=output_dir,
        run_config=run_config,
        true_params=true_params,
        plot_every_n_realizations=mc_cfg.plot_every_n_realizations,
        visualize=make_visualize_hook(true_params, mc_cfg.max_steps),
        repo_dir=REPO_ROOT,
    )
    results = runner.run()
    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"\nDone. ran={len(results)} ok={n_ok} failed={len(results) - n_ok}  output={output_dir}")


def _purge_if_config_changed(output_dir: Path, run_config: dict[str, Any]) -> None:
    """Clear stale result artifacts when the effective config hash changes.

    Prevents silently mixing realisations produced under different configurations.
    Resume within the *same* configuration is preserved.
    """
    config_path = output_dir / "mc_config.json"
    if not config_path.exists():
        return
    try:
        import json

        prior = json.loads(config_path.read_text(encoding="utf-8"))
        prior_hash = prior.get("config_hash")
    except (OSError, ValueError):
        prior_hash = None
    if prior_hash == config_hash(run_config):
        return
    warnings.warn(
        f"Config changed since last run in {output_dir}; clearing stale results.",
        RuntimeWarning,
        stacklevel=2,
    )
    for name in ("mc_results.csv", "mc_manifest.json", "mc_config.json"):
        (output_dir / name).unlink(missing_ok=True)
    hist_dir = output_dir / "histories"
    if hist_dir.exists():
        for npz in hist_dir.glob("history_r*.npz"):
            npz.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
