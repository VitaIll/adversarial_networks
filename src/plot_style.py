"""Shared black-and-white plotting style for paper-ready figures."""

from __future__ import annotations

from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

SINGLE_COLUMN_WIDTH_IN = 3.5
DOUBLE_COLUMN_WIDTH_IN = 7.0
DEFAULT_PANEL_ASPECT = 0.7

AXIS_LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 9
ANNOTATION_FONTSIZE = 9

REFERENCE_LINEWIDTH = 0.8
ESTIMATE_LINEWIDTH = 1.1

LINE_STYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "x")
HATCH_PATTERNS = ("////", "\\\\\\\\", "xxxx", "....")

_STYLE_APPLIED = False


def apply_paper_style() -> None:
    """Apply global monochrome matplotlib style once per process."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.format": "png",
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "Latin Modern Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.titleweight": "normal",
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.framealpha": 0.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "0.6",
            "grid.linestyle": ":",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    _STYLE_APPLIED = True


def figure_size(
    layout: Literal["single", "double"],
    *,
    n_rows: int = 1,
    n_cols: int = 1,
    panel_aspect: float = DEFAULT_PANEL_ASPECT,
) -> tuple[float, float]:
    """Compute figure size so each panel is within the target aspect ratio."""
    if layout not in {"single", "double"}:
        raise ValueError(f"layout must be 'single' or 'double', got {layout!r}")
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"n_rows and n_cols must be positive, got {n_rows}, {n_cols}")
    if not (0.6 <= panel_aspect <= 0.8):
        raise ValueError(
            f"panel_aspect must be in [0.6, 0.8] for paper styling, got {panel_aspect}"
        )

    width = SINGLE_COLUMN_WIDTH_IN if layout == "single" else DOUBLE_COLUMN_WIDTH_IN
    height = panel_aspect * width * (n_rows / n_cols)
    return (width, height)


def style_axes(ax: Axes) -> None:
    """Apply axis-level styling for monochrome scientific plots."""
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.4, color="0.6")
    ax.grid(False, which="minor")


def line_style(
    index: int,
    *,
    linewidth: float = ESTIMATE_LINEWIDTH,
    with_markers: bool = True,
) -> dict[str, object]:
    """Return deterministic monochrome line style for a series index."""
    style: dict[str, object] = {
        "color": "black",
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "linewidth": linewidth,
    }
    if with_markers:
        style.update(
            {
                "marker": MARKERS[index % len(MARKERS)],
                "markersize": 3.2,
                "markerfacecolor": "white",
                "markeredgewidth": 0.7,
            }
        )
    return style


def series_style(index: int, *, linewidth: float = ESTIMATE_LINEWIDTH) -> dict[str, object]:
    """Backward-compatible alias for line style with markers."""
    return line_style(index, linewidth=linewidth, with_markers=True)


def scatter_style(index: int) -> dict[str, object]:
    """Return deterministic monochrome scatter style for a series index."""
    return {
        "marker": MARKERS[index % len(MARKERS)],
        "facecolors": "white",
        "edgecolors": "black",
        "linewidths": 0.6,
        "alpha": 0.8,
    }


def add_reference_line(
    ax: Axes,
    *,
    value: float,
    text: str,
    axis: Literal["x", "y"] = "y",
) -> None:
    """Draw a dashed black reference line and annotate it at the right/top end."""
    if axis == "y":
        ax.axhline(value, color="black", linestyle="--", linewidth=REFERENCE_LINEWIDTH)
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        ax.annotate(
            text,
            xy=(0.985, value),
            xycoords=transform,
            xytext=(0, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
            clip_on=True,
        )
        return

    if axis == "x":
        ax.axvline(value, color="black", linestyle="--", linewidth=REFERENCE_LINEWIDTH)
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.annotate(
            text,
            xy=(value, 0.985),
            xycoords=transform,
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            rotation=90,
            fontsize=ANNOTATION_FONTSIZE,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
            clip_on=True,
        )
        return

    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def style_histogram_patches(
    patches: list, *, hatch_index: int, facecolor: str = "white"
) -> None:
    """Apply monochrome hatch styling to histogram bars."""
    hatch = HATCH_PATTERNS[hatch_index % len(HATCH_PATTERNS)]
    for patch in patches:
        patch.set_facecolor(facecolor)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
        patch.set_hatch(hatch)
