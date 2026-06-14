"""Visualization utilities for experiment results.

Consistent plotting functions for parameter trajectories, loss curves,
and diagnostic figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    # Package import path (e.g. `from src import visualization`)
    from .constants import (
        FIGURE_DPI,
        OPTIMAL_DISC_LOSS,
        OPTIMAL_GEN_LOSS,
        ROLLING_WINDOW_SIZE,
    )
    from .plot_style import (
        add_reference_line,
        apply_paper_style,
        figure_size,
        line_style,
        series_style,
        style_axes,
        style_histogram_patches,
    )
except ImportError:
    # Notebook/local import path when `src/` is added to sys.path.
    from constants import (  # type: ignore
        FIGURE_DPI,
        OPTIMAL_DISC_LOSS,
        OPTIMAL_GEN_LOSS,
        ROLLING_WINDOW_SIZE,
    )
    from plot_style import (  # type: ignore
        add_reference_line,
        apply_paper_style,
        figure_size,
        line_style,
        series_style,
        style_axes,
        style_histogram_patches,
    )

GRAYSCALE_CMAPS = {
    "binary",
    "bone",
    "gist_gray",
    "gray",
    "grey",
    "Greys",
}

# Ensure global style is configured before any figures are created.
apply_paper_style()


def _validate_png_path(save_path: Path) -> None:
    """Enforce PNG output for figure files."""
    if save_path.suffix.lower() != ".png":
        raise ValueError(f"save_path must end with .png, got {save_path}")


def _validate_grayscale_cmap(cmap: str) -> None:
    """Require grayscale colormaps to keep plots monochrome."""
    if cmap not in GRAYSCALE_CMAPS:
        raise ValueError(
            f"cmap must be grayscale ({sorted(GRAYSCALE_CMAPS)}), got {cmap!r}"
        )


def rolling_mean(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean of a time series.

    Args:
        values: Time series values.
        window: Window size for rolling mean.

    Returns:
        Tuple of (x_axis, smoothed_values) where x_axis starts at `window`.

    Raises:
        ValueError: If window is larger than series length.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < window:
        # Return unsmoothed if not enough data
        return np.arange(1, arr.size + 1), arr

    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth = np.convolve(arr, kernel, mode="valid")
    x_axis = np.arange(window, arr.size + 1)
    return x_axis, smooth


def plot_parameter_convergence(
    history: dict[str, list[float]],
    true_params: dict[str, float],
    save_path: Path,
    n_steps: int,
) -> None:
    """Plot parameter trajectories vs. true values.

    Args:
        history: Parameter history with keys 'beta', 'gamma', 'sigma_sq'.
        true_params: True parameter values with same keys.
        save_path: Output file path for figure.
        n_steps: Total number of training steps.

    Raises:
        ValueError: If history and true_params have mismatched keys.
    """
    apply_paper_style()
    _validate_png_path(save_path)

    required_keys = {"beta", "gamma", "sigma_sq"}
    if not required_keys.issubset(history.keys()):
        raise ValueError(f"history missing required keys: {required_keys - history.keys()}")
    if not required_keys.issubset(true_params.keys()):
        raise ValueError(
            f"true_params missing required keys: {required_keys - true_params.keys()}"
        )

    steps = np.arange(1, n_steps + 1)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figure_size("single", n_rows=3, n_cols=1),
        sharex=True,
        constrained_layout=True,
    )
    series = [
        ("beta", r"$\beta$"),
        ("gamma", r"$\gamma$"),
        ("sigma_sq", r"$\sigma^2$"),
    ]
    markevery = max(1, n_steps // 18)

    for ax, (key, label) in zip(axes, series):
        ax.plot(steps, history[key], markevery=markevery, **series_style(0))
        add_reference_line(
            ax,
            value=true_params[key],
            text=f"true={true_params[key]:.3g}",
        )
        ax.set_ylabel(label)
        style_axes(ax)

    axes[-1].set_xlabel("Generator Step")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, format="png")
    plt.close(fig)


def plot_loss_convergence(
    loss_d_history: list[float],
    loss_g_history: list[float],
    save_path: Path,
    window: int = ROLLING_WINDOW_SIZE,
) -> None:
    """Plot smoothed discriminator and generator loss curves.

    Args:
        loss_d_history: Discriminator loss history.
        loss_g_history: Generator loss history.
        save_path: Output file path for figure.
        window: Rolling mean window size.

    Raises:
        ValueError: If histories have different lengths.
    """
    apply_paper_style()
    _validate_png_path(save_path)

    if len(loss_d_history) != len(loss_g_history):
        raise ValueError(
            f"Loss histories must have same length: "
            f"D={len(loss_d_history)}, G={len(loss_g_history)}"
        )

    x_d, smooth_d = rolling_mean(loss_d_history, window=window)
    x_g, smooth_g = rolling_mean(loss_g_history, window=window)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size("double", n_rows=1, n_cols=2),
        constrained_layout=True,
    )

    axes[0].plot(x_d, smooth_d, markevery=max(1, len(x_d) // 16), **series_style(0))
    add_reference_line(axes[0], value=OPTIMAL_DISC_LOSS, text=r"$2\log 2$")
    axes[0].set_title("(a) Discriminator Loss")
    axes[0].set_xlabel("Generator Step")
    axes[0].set_ylabel(f"Rolling Mean (window={window})")
    style_axes(axes[0])

    axes[1].plot(x_g, smooth_g, markevery=max(1, len(x_g) // 16), **series_style(1))
    add_reference_line(axes[1], value=OPTIMAL_GEN_LOSS, text=r"$\log 2$")
    axes[1].set_title("(b) Generator Loss")
    axes[1].set_xlabel("Generator Step")
    axes[1].set_ylabel(f"Rolling Mean (window={window})")
    style_axes(axes[1])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, format="png")
    plt.close(fig)


def plot_tail_stability(
    history: dict[str, list[float]],
    true_params: dict[str, float],
    save_path: Path,
    tail_window: int,
) -> None:
    """Plot zoomed view of parameter trajectories over final steps.

    Args:
        history: Parameter history with keys 'beta', 'gamma', 'sigma_sq'.
        true_params: True parameter values.
        save_path: Output file path for figure.
        tail_window: Number of final steps to display.

    Raises:
        ValueError: If tail_window exceeds history length.
    """
    apply_paper_style()
    _validate_png_path(save_path)

    n_steps = len(history["beta"])
    if tail_window > n_steps:
        raise ValueError(
            f"tail_window ({tail_window}) exceeds history length ({n_steps})"
        )

    tail_steps = np.arange(n_steps - tail_window + 1, n_steps + 1)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=figure_size("single", n_rows=3, n_cols=1),
        sharex=True,
        constrained_layout=True,
    )
    series = [
        ("beta", r"$\beta$"),
        ("gamma", r"$\gamma$"),
        ("sigma_sq", r"$\sigma^2$"),
    ]
    markevery = max(1, tail_window // 18)

    for ax, (key, label) in zip(axes, series):
        ax.plot(
            tail_steps,
            history[key][-tail_window:],
            markevery=markevery,
            **series_style(0),
        )
        add_reference_line(
            ax,
            value=true_params[key],
            text=f"true={true_params[key]:.3g}",
        )
        ax.set_ylabel(label)
        style_axes(ax)

    axes[-1].set_xlabel("Generator Step")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, format="png")
    plt.close(fig)


def plot_ground_truth_graph_outcomes(
    graph: nx.Graph,
    outcomes: Sequence[float] | np.ndarray,
    save_path: Path,
    max_nodes: int | None = 1200,
    layout_seed: int = 42,
    cmap: str = "Greys",
) -> None:
    """Plot generated ground-truth graph with nodes shaded by outcome values.

    Args:
        graph: Undirected graph used for synthetic data generation.
        outcomes: Outcome values aligned with ``list(graph.nodes())`` order.
        save_path: Output file path for figure.
        max_nodes: Optional cap on plotted nodes for readability/performance.
            If exceeded, a deterministic node subset is drawn.
        layout_seed: Seed used for deterministic layout and optional subsampling.
        cmap: Matplotlib grayscale colormap.

    Raises:
        ValueError: If graph is empty, outcomes are malformed, lengths mismatch,
            or colormap is not grayscale.
    """
    apply_paper_style()
    _validate_png_path(save_path)
    _validate_grayscale_cmap(cmap)

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot plot an empty graph.")

    node_order = list(graph.nodes())
    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"outcomes must be 1D, got shape {y.shape}")
    if y.size != len(node_order):
        raise ValueError(
            f"outcomes length ({y.size}) must match number of nodes ({len(node_order)})"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("outcomes contain non-finite values.")

    if max_nodes is not None and max_nodes <= 0:
        raise ValueError(f"max_nodes must be positive or None, got {max_nodes}")

    rng = np.random.default_rng(layout_seed)
    if max_nodes is not None and len(node_order) > max_nodes:
        sampled_idx = np.sort(rng.choice(len(node_order), size=max_nodes, replace=False))
        plot_nodes = [node_order[i] for i in sampled_idx]
    else:
        sampled_idx = np.arange(len(node_order))
        plot_nodes = node_order

    graph_plot = graph.subgraph(plot_nodes).copy()
    y_plot = y[sampled_idx]

    n_plot = graph_plot.number_of_nodes()
    iterations = 30 if n_plot > 1000 else 60
    pos = nx.spring_layout(graph_plot, seed=layout_seed, iterations=iterations)

    fig, ax = plt.subplots(
        figsize=figure_size("double", n_rows=1, n_cols=1),
        constrained_layout=True,
    )
    nx.draw_networkx_edges(
        graph_plot,
        pos=pos,
        ax=ax,
        width=0.25,
        alpha=0.12,
        edge_color="0.4",
    )

    vmin = float(np.min(y_plot))
    vmax = float(np.max(y_plot))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    node_artist = nx.draw_networkx_nodes(
        graph_plot,
        pos=pos,
        ax=ax,
        node_size=14 if n_plot > 1000 else 22,
        node_color=y_plot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.2,
        edgecolors="black",
        alpha=0.95,
    )

    colorbar = fig.colorbar(node_artist, ax=ax, fraction=0.05, pad=0.01)
    colorbar.set_label("Outcome Value (Y_obs)")
    colorbar.ax.tick_params(axis="y", which="both", direction="out")

    sampled = n_plot != graph.number_of_nodes()
    if sampled:
        ax.text(
            0.01,
            0.02,
            f"sampled {n_plot}/{graph.number_of_nodes()} nodes",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
        )

    ax.set_axis_off()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, format="png")
    plt.close(fig)


def _validate_quantiles(quantiles: tuple[float, ...]) -> None:
    """Validate quantile tuple for envelope plotting."""
    if len(quantiles) != 5:
        raise ValueError(f"quantiles must contain 5 values, got {len(quantiles)}")
    if any((quantile < 0.0 or quantile > 1.0) for quantile in quantiles):
        raise ValueError("quantiles must be in [0, 1].")
    if tuple(sorted(quantiles)) != quantiles:
        raise ValueError("quantiles must be sorted in ascending order.")


def _pad_history(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad/truncate one history array to target length using edge padding."""
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    values = np.asarray(arr, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("History arrays must be non-empty.")
    if values.size < target_len:
        pad_width = target_len - values.size
        return np.pad(values, (0, pad_width), mode="edge")
    return values[:target_len]


def _trailing_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute trailing moving average with min-periods=1 and fixed output length."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("values cannot be empty.")

    cumsum = np.cumsum(arr, dtype=np.float64)
    out = np.empty_like(arr)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        out[idx] = total / float(idx - start + 1)
    return out


_PARAM_SYMBOLS = {
    "beta": r"\beta", "gamma": r"\gamma", "sigma_sq": r"\sigma^2",
    "lambda_": r"\lambda", "mu": r"\mu", "r": "r", "alpha": r"\alpha",
}


def _error_label(name: str) -> str:
    """A monochrome error-axis label ``\\hat{sym} - sym_0`` for a parameter name."""
    sym = _PARAM_SYMBOLS.get(name)
    if sym is None:
        return f"{name} error"
    return rf"$\hat{{{sym}}}-{sym}_0$"


def plot_mc_parameter_distributions(
    results: list[dict],
    true_params: dict[str, float],
    save_path: Path,
    n_bins: int = 25,
    series: Sequence[str] | None = None,
) -> None:
    """Plot histograms of parameter estimation errors across MC realizations.

    Args:
        results: Per-realization result rows with ``*_hat`` entries.
        true_params: True parameter values, keyed by name.
        save_path: Output path for the PNG figure.
        n_bins: Histogram bin count.
        series: Parameter names to plot (default: the keys of ``true_params``) —
            model-agnostic, so the effort game or a custom game's parameters plot
            without code changes.

    Raises:
        ValueError: If required values are missing or invalid.
    """
    apply_paper_style()
    _validate_png_path(save_path)

    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    names = list(series) if series is not None else list(true_params.keys())
    missing_true = set(names).difference(true_params.keys())
    if missing_true:
        raise ValueError(f"true_params missing required keys: {sorted(missing_true)}")

    keys = [(f"{name}_hat", name, _error_label(name)) for name in names]

    errors: list[np.ndarray] = []
    for hat_key, true_key, _ in keys:
        vals: list[float] = []
        for row in results:
            if row.get("status") not in (None, "ok"):
                continue
            if hat_key not in row:
                continue
            estimate = float(row[hat_key])
            if np.isfinite(estimate):
                vals.append(estimate - float(true_params[true_key]))
        if not vals:
            raise ValueError(f"No finite values available for {hat_key}.")
        errors.append(np.asarray(vals, dtype=np.float64))

    fig, axes = plt.subplots(
        len(names),
        1,
        figsize=figure_size("single", n_rows=len(names), n_cols=1),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for idx, (ax, (_, _, x_label), err_values) in enumerate(zip(axes, keys, errors, strict=True)):
        _, _, patches = ax.hist(
            err_values,
            bins=n_bins,
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
        )
        style_histogram_patches(list(patches), hatch_index=idx, facecolor="white")
        add_reference_line(ax, value=0.0, text="0", axis="x")

        mean_val = float(np.mean(err_values))
        std_val = float(np.std(err_values, ddof=0))
        n_obs = int(err_values.size)
        ax.text(
            0.98,
            0.95,
            f"mean={mean_val:+.4f}\nstd={std_val:.4f}\nn={n_obs}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
        )
        ax.set_xlabel("Estimation error")
        ax.set_ylabel("Count")
        ax.set_title(x_label)
        style_axes(ax)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, format="png")
    plt.close(fig)


def plot_mc_quantile_convergence_paths(
    histories: list[dict[str, np.ndarray]],
    true_params: dict[str, float],
    max_steps: int,
    save_path: Path,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    series: Sequence[str] | None = None,
) -> None:
    """Plot parameter trajectories with Monte Carlo quantile envelopes.

    Args:
        histories: Per-realization histories loaded from ``.npz`` files.
        true_params: True values for ``beta``, ``gamma``, and ``sigma_sq``.
        max_steps: Common padded horizon for plotted trajectories.
        save_path: Output PNG path.
        quantiles: Quantiles used to draw outer and inner envelopes.

    Raises:
        ValueError: If input histories are empty or malformed.
    """
    apply_paper_style()
    _validate_png_path(save_path)
    _validate_quantiles(quantiles)

    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if not histories:
        raise ValueError("histories cannot be empty.")

    names = list(series) if series is not None else list(true_params.keys())
    missing_true = set(names).difference(true_params.keys())
    if missing_true:
        raise ValueError(f"true_params missing required keys: {sorted(missing_true)}")

    steps = np.arange(1, max_steps + 1, dtype=np.int32)
    panels = [(name, f"${_PARAM_SYMBOLS.get(name, name)}$") for name in names]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=figure_size("single", n_rows=len(panels), n_cols=1),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for ax, (key, ylabel) in zip(axes, panels, strict=True):
        stacked = np.stack(
            [_pad_history(np.asarray(hist[key]), max_steps) for hist in histories],
            axis=0,
        )
        q_lo, q_iqr_lo, q_med, q_iqr_hi, q_hi = np.quantile(
            stacked,
            q=np.asarray(quantiles, dtype=np.float64),
            axis=0,
        )

        ax.fill_between(steps, q_lo, q_hi, color="0.7", alpha=0.15, linewidth=0.0)
        ax.fill_between(steps, q_iqr_lo, q_iqr_hi, color="0.45", alpha=0.35, linewidth=0.0)
        ax.plot(steps, q_med, **line_style(0, with_markers=False))
        add_reference_line(ax, value=float(true_params[key]), text=f"true={true_params[key]:.3g}")
        ax.set_ylabel(ylabel)
        style_axes(ax)

    axes[-1].set_xlabel("Generator Step")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, format="png")
    plt.close(fig)


def plot_mc_quantile_loss_paths(
    histories: list[dict[str, np.ndarray]],
    max_steps: int,
    save_path: Path,
    smoothing_window: int = 30,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> None:
    """Plot discriminator/generator loss quantile envelopes across realizations.

    Args:
        histories: Per-realization histories loaded from ``.npz`` files.
        max_steps: Common padded horizon for plotted trajectories.
        save_path: Output PNG path.
        smoothing_window: Trailing rolling-average window applied before quantiles.
        quantiles: Quantiles used to draw outer and inner envelopes.

    Raises:
        ValueError: If input histories are empty or malformed.
    """
    apply_paper_style()
    _validate_png_path(save_path)
    _validate_quantiles(quantiles)

    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if smoothing_window <= 0:
        raise ValueError(f"smoothing_window must be positive, got {smoothing_window}")
    if not histories:
        raise ValueError("histories cannot be empty.")

    steps = np.arange(1, max_steps + 1, dtype=np.int32)
    loss_specs = [
        ("loss_d", "(a) Discriminator Loss", OPTIMAL_DISC_LOSS, r"$2\log 2$"),
        ("loss_g", "(b) Generator Loss", OPTIMAL_GEN_LOSS, r"$\log 2$"),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figure_size("double", n_rows=1, n_cols=2),
        constrained_layout=True,
    )

    for ax, (key, title, ref_val, ref_text) in zip(axes, loss_specs, strict=True):
        stacked_raw = np.stack(
            [_pad_history(np.asarray(hist[key]), max_steps) for hist in histories],
            axis=0,
        )
        stacked_smooth = np.stack(
            [_trailing_rolling_mean(row, window=smoothing_window) for row in stacked_raw],
            axis=0,
        )
        q_lo, q_iqr_lo, q_med, q_iqr_hi, q_hi = np.quantile(
            stacked_smooth,
            q=np.asarray(quantiles, dtype=np.float64),
            axis=0,
        )

        ax.fill_between(steps, q_lo, q_hi, color="0.7", alpha=0.15, linewidth=0.0)
        ax.fill_between(steps, q_iqr_lo, q_iqr_hi, color="0.45", alpha=0.35, linewidth=0.0)
        ax.plot(steps, q_med, **line_style(0, with_markers=False))
        add_reference_line(ax, value=ref_val, text=ref_text)
        ax.set_title(title)
        ax.set_xlabel("Generator Step")
        ax.set_ylabel("Loss")
        style_axes(ax)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=FIGURE_DPI, format="png")
    plt.close(fig)
