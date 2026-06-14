"""Monte Carlo orchestration over the adversarial estimation engine.

:class:`MonteCarloRunner` drives many independent estimation realisations on a
*shared* :class:`~src.ego.EgoSubstrate`, handling everything that is not the
estimation itself: deterministic per-realisation seed streams, resume from a
partial run, durable per-realisation logging (CSV rows + per-step ``.npz``),
provenance (config hash, library versions, git SHA), and periodic visualisation.

The runner is pure orchestration and model-agnostic: the model-specific steps
(simulating the observed outcome from a true model, and constructing the fresh
model + discriminator + config with sampled initial parameters) are injected as
callables. The same runner therefore serves the linear-in-means Monte Carlo, the
effort game, and any future network game.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from .contracts import EstimationResult, MetricsObserver
from .core.types import InstanceNoiseConfigLike
from .ego import EgoSubstrate
from .estimator import GradientTransform, _run_minimax
from .estimator_config import EstimatorConfig
from .io_utils import append_realization_row, load_completed_realizations, save_json_manifest
from .observability import InMemoryHistory
from .provenance import collect_provenance


@dataclass
class RealizationSpec:
    """The per-realisation pieces the runner feeds to :func:`_run_minimax`.

    The model-specific work (a fresh model + discriminator + config with the sampled
    initial parameters) is built by the ``estimator_factory``; the runner owns the
    shared substrate, the observed outcome, logging and provenance.
    """

    model: object
    discriminator: object
    config: EstimatorConfig
    init_params: Mapping[str, float]
    instance_noise: InstanceNoiseConfigLike | None = None
    gradient_transform: GradientTransform | None = None
    observers: tuple[MetricsObserver, ...] = ()


# (substrate, gt_seed) -> observed outcome vector Y_obs
ObservedFactory = Callable[[EgoSubstrate, int], Tensor]
# (substrate, Y_obs, realization_idx, seeds) -> RealizationSpec
EstimatorFactory = Callable[[EgoSubstrate, Tensor, int, Mapping[str, int]], "RealizationSpec"]
# (output_dir, completed realisation results) -> None
VisualizeHook = Callable[[Path, "list[RealizationResult]"], None]


def derive_seed(master_seed: int, realization_idx: int, stream_id: int) -> int:
    """Derive an independent uint32 seed from a master seed and stream id.

    Uses ``numpy.random.SeedSequence`` so distinct ``(realization, stream)`` pairs
    yield statistically independent seeds. Stream ids: 0=ground-truth, 1=training,
    2=sampler, 3=initialisation (matching the original experiment script).
    """
    seq = np.random.SeedSequence([int(master_seed), int(realization_idx), int(stream_id)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def seed_streams(master_seed: int, realization_idx: int) -> dict[str, int]:
    """Return the four named per-realisation seeds."""
    return {
        "gt": derive_seed(master_seed, realization_idx, 0),
        "train": derive_seed(master_seed, realization_idx, 1),
        "sampler": derive_seed(master_seed, realization_idx, 2),
        "init": derive_seed(master_seed, realization_idx, 3),
    }


@dataclass(frozen=True)
class RealizationResult:
    """One Monte Carlo realisation outcome, ready to flatten to a CSV row.

    Attributes:
        realization: Realisation index.
        result: The engine's :class:`~src.contracts.EstimationResult`.
        seeds: The four derived seeds for this realisation.
        init_params: The sampled initial structural parameters.
        elapsed_seconds: Wall-clock time for the realisation.
    """

    realization: int
    result: EstimationResult
    seeds: Mapping[str, int]
    init_params: Mapping[str, float]
    elapsed_seconds: float

    @property
    def status(self) -> str:
        return self.result.status

    def to_row(self) -> dict[str, Any]:
        """Flatten to a CSV-friendly row with per-parameter ``*_hat``/``*_final`` columns."""
        row: dict[str, Any] = {
            "realization": int(self.realization),
            "converged": bool(self.result.converged),
            "final_step": int(self.result.final_step),
            "status": self.result.status,
        }
        for name, value in self.result.params.items():
            row[f"{name}_hat"] = float(value)
        for name, value in self.result.params_final.items():
            row[f"{name}_final"] = float(value)
        row["loss_d_rolling_final"] = float(self.result.loss_d_rolling_final)
        row["loss_g_rolling_final"] = float(self.result.loss_g_rolling_final)
        for name, value in self.init_params.items():
            row[f"{name}_init"] = float(value)
        for stream, seed in self.seeds.items():
            row[f"{stream}_seed"] = int(seed)
        row["elapsed_seconds"] = round(float(self.elapsed_seconds), 3)
        return row


@dataclass
class MonteCarloRunner:
    """Orchestrates repeated estimation realisations over a shared substrate.

    Args:
        substrate: The shared :class:`~src.ego.EgoSubstrate` (built once).
        n_realizations: Number of realisations to run.
        master_seed: Master seed for the per-realisation seed streams.
        observed_factory: Builds the observed outcome ``Y_obs`` for a realisation.
        estimator_factory: Builds ``(estimator, init_params)`` for a realisation.
        output_dir: Directory for ``mc_results.csv``, ``mc_manifest.json``,
            ``mc_config.json`` and the ``histories/`` ``.npz`` files.
        run_config: Effective configuration dict recorded in the manifest/hash.
        true_params: Optional ground-truth parameters for error summaries.
        plot_every_n_realizations: Visualisation cadence (``0`` disables periodic
            refresh).
        visualize: Optional hook to (re)generate figures from current results.
        repo_dir: Directory used for the git-SHA provenance lookup.
    """

    substrate: EgoSubstrate
    n_realizations: int
    master_seed: int
    observed_factory: ObservedFactory
    estimator_factory: EstimatorFactory
    output_dir: Path
    run_config: Mapping[str, Any] = field(default_factory=dict)
    true_params: Mapping[str, float] | None = None
    plot_every_n_realizations: int = 0
    visualize: VisualizeHook | None = None
    repo_dir: str | Path | None = None

    def run(self) -> list[RealizationResult]:
        """Run all (remaining) realisations, returning their results.

        Resumes from any realisations already present in ``mc_results.csv``. Writes
        each realisation's row and per-step history as it completes, and a final
        provenance + summary manifest.
        """
        if self.n_realizations <= 0:
            raise ValueError(f"n_realizations must be positive, got {self.n_realizations}")
        output_dir = Path(self.output_dir)
        histories_dir = output_dir / "histories"
        output_dir.mkdir(parents=True, exist_ok=True)
        histories_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "mc_results.csv"
        manifest_path = output_dir / "mc_manifest.json"
        config_path = output_dir / "mc_config.json"

        provenance = collect_provenance(self.run_config, cwd=self.repo_dir, extra={"run_config": dict(self.run_config)})
        save_json_manifest(config_path, provenance)

        completed_rows = load_completed_realizations(results_path)
        completed = {int(row["realization"]) for row in completed_rows if "realization" in row}

        results: list[RealizationResult] = []
        run_start = time.time()
        for idx in range(self.n_realizations):
            if idx in completed:
                continue
            seeds = seed_streams(self.master_seed, idx)
            Y_obs = self.observed_factory(self.substrate, seeds["gt"])
            spec = self.estimator_factory(self.substrate, Y_obs, idx, seeds)

            history = InMemoryHistory()
            observers = list(spec.observers) + [history]

            t0 = time.time()
            engine_result = _run_minimax(
                model=spec.model, discriminator=spec.discriminator, substrate=self.substrate,
                Y_obs=Y_obs, config=spec.config, instance_noise=spec.instance_noise,
                observers=observers, gradient_transform=spec.gradient_transform,
            )
            elapsed = time.time() - t0

            realization_result = RealizationResult(
                realization=idx, result=engine_result, seeds=seeds,
                init_params=spec.init_params, elapsed_seconds=elapsed,
            )
            results.append(realization_result)

            if len(history) > 0:
                arrays = history.as_arrays()
                np.savez_compressed(histories_dir / f"history_r{idx:04d}.npz", **arrays)
            append_realization_row(results_path, realization_result.to_row())

            if (
                self.visualize is not None
                and self.plot_every_n_realizations > 0
                and (len(completed) + len(results)) % self.plot_every_n_realizations == 0
            ):
                self.visualize(output_dir, results)

        if self.visualize is not None and results:
            self.visualize(output_dir, results)

        manifest = {
            **provenance,
            "runtime_seconds": round(time.time() - run_start, 2),
            "n_requested": int(self.n_realizations),
            "n_completed_this_run": len(results),
            "n_previously_completed": len(completed),
            "n_failed_this_run": sum(1 for r in results if r.status != "ok"),
            "parameter_summary": self._summarize(results),
        }
        save_json_manifest(manifest_path, manifest)
        return results

    def _summarize(self, results: list[RealizationResult]) -> dict[str, dict[str, float]]:
        """Per-parameter mean/std (+ bias/rmse against ``true_params`` if given)."""
        ok = [r for r in results if r.status == "ok"]
        if not ok:
            return {}
        param_names = list(ok[0].result.params.keys())
        summary: dict[str, dict[str, float]] = {}
        for name in param_names:
            values = np.asarray(
                [r.result.params[name] for r in ok if np.isfinite(r.result.params[name])],
                dtype=np.float64,
            )
            if values.size == 0:
                continue
            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "median": float(np.median(values)),
                "count": int(values.size),
            }
            if self.true_params is not None and name in self.true_params:
                truth = float(self.true_params[name])
                errors = values - truth
                stats["true"] = truth
                stats["bias"] = float(np.mean(errors))
                stats["rmse"] = float(np.sqrt(np.mean(np.square(errors))))
            summary[name] = stats
        return summary
