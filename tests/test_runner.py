"""Tests for the MonteCarloRunner orchestration layer.

Covers deterministic/independent seed streams and a small end-to-end run over a
*shared* substrate: durable CSV rows (model-agnostic ``*_hat`` columns), per-step
``.npz`` histories, the provenance manifest, the model-agnostic parameter summary,
and resume from a partial run. The runner drives :func:`_run_minimax` directly via
a per-realisation :class:`RealizationSpec`.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from adversarial_networks.discriminator import RootedMPNNDiscriminator
from adversarial_networks.ego import EgoSubstrate
from adversarial_networks.estimator_config import EstimatorConfig
from adversarial_networks.generators import LinearInMeansGenerator
from adversarial_networks.runner import (
    MonteCarloRunner,
    RealizationResult,
    RealizationSpec,
    seed_streams,
)


def test_seed_streams_deterministic_and_independent() -> None:
    a = seed_streams(42, 0)
    assert a == seed_streams(42, 0)
    assert seed_streams(42, 1) != a
    assert seed_streams(7, 0) != a
    assert len(set(a.values())) == 4


def _substrate(n: int = 30, seed: int = 0) -> EgoSubstrate:
    graph = nx.barabasi_albert_graph(n=n, m=2, seed=seed)
    torch.manual_seed(seed)
    X = torch.randn(graph.number_of_nodes())
    return EgoSubstrate.from_networkx(graph, X, k=2, root_sampler_mode="uniform", seed=seed)


def _observed(substrate: EgoSubstrate, gt_seed: int) -> torch.Tensor:
    torch.manual_seed(gt_seed)
    true_model = LinearInMeansGenerator(
        beta_cap=0.85, picard_tol=1e-6, picard_max=40, init_beta=0.4, init_gamma=1.5,
    )
    with torch.no_grad():
        return true_model(substrate.W, substrate.X)


def _factory(substrate, Y_obs, idx, seeds) -> RealizationSpec:
    rng = np.random.default_rng(seeds["init"])
    init_beta = float(rng.uniform(0.0, 0.4))
    init_gamma = float(rng.uniform(0.0, 0.5))
    model = LinearInMeansGenerator(
        beta_cap=0.85, picard_tol=1e-6, picard_max=40, init_beta=init_beta, init_gamma=init_gamma,
    )
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    config = EstimatorConfig(max_steps=8, batch_size=8, n_disc=1, lr_d=1e-3, lr_g=5e-3, seed=seeds["train"])
    return RealizationSpec(
        model=model, discriminator=disc, config=config,
        init_params={"beta": init_beta, "gamma": init_gamma, "log_sigma_sq": 0.0},
    )


def _runner(output_dir: Path, n: int, substrate: EgoSubstrate) -> MonteCarloRunner:
    return MonteCarloRunner(
        substrate=substrate, n_realizations=n, master_seed=42,
        observed_factory=_observed, estimator_factory=_factory, output_dir=output_dir,
        run_config={"model": "LinearInMeansGenerator", "max_steps": 8},
        true_params={"beta": 0.4, "gamma": 1.5, "sigma_sq": 1.0},
    )


def test_runner_end_to_end_writes_artifacts(tmp_path: Path) -> None:
    substrate = _substrate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = _runner(tmp_path, n=2, substrate=substrate).run()

    assert len(results) == 2
    assert all(isinstance(r, RealizationResult) for r in results)
    assert all(r.status == "ok" for r in results)

    with (tmp_path / "mc_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {"beta_hat", "gamma_hat", "sigma_sq_hat", "converged", "train_seed"} <= set(rows[0].keys())

    for idx in range(2):
        npz_path = tmp_path / "histories" / f"history_r{idx:04d}.npz"
        assert npz_path.exists()
        with np.load(npz_path) as arr:
            assert {"beta", "gamma", "sigma_sq", "loss_d", "loss_g"} <= set(arr.files)
            assert arr["beta"].shape[0] == 8

    manifest = json.loads((tmp_path / "mc_manifest.json").read_text(encoding="utf-8"))
    assert "config_hash" in manifest and "versions" in manifest
    assert set(manifest["parameter_summary"].keys()) == {"beta", "gamma", "sigma_sq"}
    assert math.isfinite(manifest["parameter_summary"]["gamma"]["rmse"])


def test_runner_resumes_from_partial_run(tmp_path: Path) -> None:
    substrate = _substrate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = _runner(tmp_path, n=1, substrate=substrate).run()
        assert len(first) == 1 and first[0].realization == 0
        second = _runner(tmp_path, n=2, substrate=substrate).run()
    assert len(second) == 1
    assert second[0].realization == 1

    with (tmp_path / "mc_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {int(r["realization"]) for r in rows} == {0, 1}
