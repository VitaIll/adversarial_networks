"""Visualization utilities for experiment results.

Consistent plotting functions for parameter trajectories, loss curves,
and diagnostic figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
        series_style,
        style_axes,
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
        series_style,
        style_axes,
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
