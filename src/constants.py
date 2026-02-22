"""Named constants used across the adversarial networks MVP.

All magic numbers are defined here for maintainability and clarity.
"""

from __future__ import annotations

import math

# ─── Numerical Tolerances ───
DEFAULT_PICARD_TOL = 1e-6
"""Default convergence tolerance for Picard iteration."""

DEFAULT_ATOL = 1e-6
"""Default absolute tolerance for numerical comparisons."""

# ─── Feature Dimensions ───
NUM_NODE_FEATURES = 3
"""Number of node features: [X_tilde, Y_tilde, root_marker]."""

ROOT_MARKER_INDEX = 2
"""Index of the root marker in node feature vectors."""

# ─── Visualization Constants ───
FIGURE_DPI = 150
"""DPI for saved figures."""

ROLLING_WINDOW_SIZE = 120
"""Window size for rolling mean smoothing of loss curves."""

# ─── GAN Equilibrium Targets ───
OPTIMAL_DISC_LOSS = 2.0 * math.log(2.0)
"""Theoretical optimal discriminator loss at Nash equilibrium ≈ 1.386."""

OPTIMAL_GEN_LOSS = math.log(2.0)
"""Theoretical optimal generator loss at Nash equilibrium ≈ 0.693."""

# ─── File Extensions ───
FIGURE_EXTENSION = ".png"
"""Extension for saved figures."""

TABLE_EXTENSION = ".csv"
"""Extension for saved tables."""

MANIFEST_FILENAME = "run_manifest.json"
"""Filename for run metadata manifest."""

FIG_GROUND_TRUTH_GRAPH_OUTCOMES = "fig07_ground_truth_graph_outcomes.png"
"""Filename for graph plot with node colors set by observed outcomes."""
