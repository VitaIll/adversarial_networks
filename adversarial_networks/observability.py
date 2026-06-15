"""Observability sinks for the adversarial estimation engine.

The engine emits one :class:`~adversarial_networks.contracts.StepMetrics` per outer
step to a list of :class:`~adversarial_networks.contracts.MetricsObserver` sinks.
This module provides the
concrete sinks:

* :class:`InMemoryHistory` — collects every step into arrays (the structured
  replacement for the ad-hoc ``history`` dict in the original script), and exposes
  :meth:`InMemoryHistory.as_arrays` for the Monte Carlo ``.npz`` and quantile
  plots.
* :class:`ConsoleLogger` — periodic human-readable progress, generalised over
  whatever parameters the model reports.
* :class:`JsonlSink` — streams one JSON record per step to disk for durable,
  inspectable run logs.
* :class:`CompositeObserver` — fans a single event out to several sinks.

Sinks are intentionally simple and side-effect isolated; the engine guards
observer dispatch so a misbehaving sink degrades observability without corrupting
the estimation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from .contracts import EstimationResult, MetricsObserver, StepMetrics


class InMemoryHistory(MetricsObserver):
    """Collects the full per-step history in memory.

    The parameter set is discovered from the first observed step, so the history
    is model-agnostic. After a run, :meth:`as_arrays` returns a flat mapping
    suitable for ``numpy.savez`` and the quantile-path plots.
    """

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.params: dict[str, list[float]] = {}
        self.loss_d: list[float] = []
        self.loss_g: list[float] = []
        self.loss_d_rolling: list[float] = []
        self.loss_g_rolling: list[float] = []
        self.grad_norm_g: list[float] = []
        self.tau_y: list[float] = []
        self.picard_iterations: list[int] = []
        self.result: EstimationResult | None = None

    def on_run_start(self, meta: Mapping[str, object]) -> None:  # noqa: D102 - see protocol
        return None

    def on_step(self, metrics: StepMetrics) -> None:  # noqa: D102 - see protocol
        if not self.params:
            self.params = {name: [] for name in metrics.params}
        for name, path in self.params.items():
            path.append(float(metrics.params[name]))
        self.steps.append(int(metrics.step))
        self.loss_d.append(float(metrics.loss_d))
        self.loss_g.append(float(metrics.loss_g))
        self.loss_d_rolling.append(float(metrics.loss_d_rolling))
        self.loss_g_rolling.append(float(metrics.loss_g_rolling))
        self.grad_norm_g.append(float(metrics.grad_norm_g))
        self.tau_y.append(float(metrics.tau_y))
        self.picard_iterations.append(int(metrics.picard_iterations))

    def on_run_end(self, result: EstimationResult) -> None:  # noqa: D102 - see protocol
        self.result = result

    def __len__(self) -> int:
        return len(self.steps)

    def param_history(self) -> dict[str, list[float]]:
        """Return a shallow copy of the per-parameter value paths."""
        return {name: list(path) for name, path in self.params.items()}

    def as_arrays(self) -> dict[str, object]:
        """Flatten the history into a mapping of 1-D float arrays.

        Keys are the model's parameter names plus ``loss_d``, ``loss_g``,
        ``loss_d_rolling``, ``loss_g_rolling``, ``grad_norm_g``, ``tau_y``.
        Returns plain numpy arrays so the caller can ``np.savez`` them.
        """
        import numpy as np

        arrays: dict[str, object] = {}
        for name, path in self.params.items():
            arrays[name] = np.asarray(path, dtype=np.float64)
        arrays["loss_d"] = np.asarray(self.loss_d, dtype=np.float64)
        arrays["loss_g"] = np.asarray(self.loss_g, dtype=np.float64)
        arrays["loss_d_rolling"] = np.asarray(self.loss_d_rolling, dtype=np.float64)
        arrays["loss_g_rolling"] = np.asarray(self.loss_g_rolling, dtype=np.float64)
        arrays["grad_norm_g"] = np.asarray(self.grad_norm_g, dtype=np.float64)
        arrays["tau_y"] = np.asarray(self.tau_y, dtype=np.float64)
        return arrays


class ConsoleLogger(MetricsObserver):
    """Prints periodic, human-readable progress lines.

    Args:
        every_n_steps: Print cadence in outer steps; ``None`` disables printing.
        prefix: Optional label (e.g. a realisation id) prepended to each line.
    """

    def __init__(self, every_n_steps: int | None = 100, prefix: str = "") -> None:
        if every_n_steps is not None and every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive when provided.")
        self._every = every_n_steps
        self._prefix = prefix

    def on_run_start(self, meta: Mapping[str, object]) -> None:  # noqa: D102 - see protocol
        return None

    def on_step(self, metrics: StepMetrics) -> None:  # noqa: D102 - see protocol
        if self._every is None or metrics.step % self._every != 0:
            return
        params = " ".join(f"{name}={value:.4f}" for name, value in metrics.params.items())
        head = f"{self._prefix} " if self._prefix else ""
        print(
            f"{head}step={metrics.step:05d} {params} "
            f"loss=({metrics.loss_d:.4f}, {metrics.loss_g:.4f}) "
            f"roll=({metrics.loss_d_rolling:.4f}, {metrics.loss_g_rolling:.4f}) "
            f"|g|={metrics.grad_norm_g:.3f}",
            flush=True,
        )

    def on_run_end(self, result: EstimationResult) -> None:  # noqa: D102 - see protocol
        params = " ".join(f"{name}={value:.4f}" for name, value in result.params.items())
        head = f"{self._prefix} " if self._prefix else ""
        print(
            f"{head}done status={result.status} converged={result.converged} "
            f"steps={result.final_step} estimate=({params})",
            flush=True,
        )


class JsonlSink(MetricsObserver):
    """Streams one JSON record per step to a JSON Lines file.

    The file is opened at run start and closed at run end. Each line is a JSON
    object of the full :class:`~adversarial_networks.contracts.StepMetrics`. A trailing record with
    the run result is written at the end.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._handle = None

    def on_run_start(self, meta: Mapping[str, object]) -> None:  # noqa: D102 - see protocol
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")
        self._handle.write(json.dumps({"event": "run_start", "meta": _jsonable(meta)}) + "\n")
        self._handle.flush()

    def on_step(self, metrics: StepMetrics) -> None:  # noqa: D102 - see protocol
        if self._handle is None:
            return
        record = {"event": "step", **asdict(metrics)}
        self._handle.write(json.dumps(record, default=_jsonable) + "\n")

    def on_run_end(self, result: EstimationResult) -> None:  # noqa: D102 - see protocol
        if self._handle is None:
            return
        self._handle.write(json.dumps({"event": "run_end", **asdict(result)}, default=_jsonable) + "\n")
        self._handle.flush()
        self._handle.close()
        self._handle = None


class CompositeObserver(MetricsObserver):
    """Fans a single observability event out to a sequence of sinks."""

    def __init__(self, observers: Sequence[MetricsObserver]) -> None:
        self._observers = list(observers)

    def on_run_start(self, meta: Mapping[str, object]) -> None:  # noqa: D102 - see protocol
        for observer in self._observers:
            observer.on_run_start(meta)

    def on_step(self, metrics: StepMetrics) -> None:  # noqa: D102 - see protocol
        for observer in self._observers:
            observer.on_step(metrics)

    def on_run_end(self, result: EstimationResult) -> None:  # noqa: D102 - see protocol
        for observer in self._observers:
            observer.on_run_end(result)


def _jsonable(value: object) -> object:
    """Best-effort conversion of values to JSON-serialisable forms."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
