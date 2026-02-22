"""Tests for visualization helpers."""

from __future__ import annotations

import matplotlib
import networkx as nx
import numpy as np
import pytest

from src.visualization import plot_ground_truth_graph_outcomes

matplotlib.use("Agg")


def test_plot_ground_truth_graph_outcomes_writes_file(tmp_path) -> None:
    """Ground-truth graph outcome plot is saved to disk."""
    graph = nx.barabasi_albert_graph(40, 2, seed=7)
    outcomes = np.linspace(-1.0, 1.0, graph.number_of_nodes(), dtype=np.float64)
    out_path = tmp_path / "fig07_ground_truth_graph_outcomes.png"

    plot_ground_truth_graph_outcomes(
        graph=graph,
        outcomes=outcomes,
        save_path=out_path,
        max_nodes=None,
        layout_seed=11,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_ground_truth_graph_outcomes_validates_length(tmp_path) -> None:
    """Mismatched outcome length raises a clear error."""
    graph = nx.path_graph(6)
    outcomes = np.zeros(5, dtype=np.float64)

    with pytest.raises(ValueError, match="must match number of nodes"):
        plot_ground_truth_graph_outcomes(
            graph=graph,
            outcomes=outcomes,
            save_path=tmp_path / "unused.png",
        )


def test_plot_ground_truth_graph_outcomes_requires_png_output(tmp_path) -> None:
    """Only PNG output paths are accepted for figure export."""
    graph = nx.path_graph(6)
    outcomes = np.zeros(graph.number_of_nodes(), dtype=np.float64)

    with pytest.raises(ValueError, match=r"must end with \.png"):
        plot_ground_truth_graph_outcomes(
            graph=graph,
            outcomes=outcomes,
            save_path=tmp_path / "unused.pdf",
        )


def test_plot_ground_truth_graph_outcomes_requires_grayscale_cmap(tmp_path) -> None:
    """Non-grayscale colormaps are rejected to enforce monochrome style."""
    graph = nx.path_graph(6)
    outcomes = np.zeros(graph.number_of_nodes(), dtype=np.float64)

    with pytest.raises(ValueError, match="grayscale"):
        plot_ground_truth_graph_outcomes(
            graph=graph,
            outcomes=outcomes,
            save_path=tmp_path / "unused.png",
            cmap="viridis",
        )
