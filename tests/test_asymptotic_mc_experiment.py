"""Coherence tests for the asymptotic Monte Carlo experiment driver's substrate build.

Pins that ``build_substrate`` accepts the graph_type the shipped reference config ships
— ``ExperimentConfig.default()`` is LFR (paper-scale), and the driver now dispatches on
``cfg.graph.graph_type`` (reusing ``datasets._build_lfr_graph`` with the GraphConfig cap
knobs) instead of hard-rejecting non-BA, so ``default()`` is actually runnable (D8-06-R2).
The paper-scale ``n_nodes`` is overridden to a small, fast LFR for the smoke test; the
graph_type (the thing that used to raise) is unchanged.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adversarial_networks.config import ExperimentConfig, MonteCarloConfig  # noqa: E402
from adversarial_networks.ego import EgoSubstrate  # noqa: E402
from experiments.asymptotic_mc_experiment import build_substrate  # noqa: E402


def test_build_substrate_accepts_shipped_default_graph_type_lfr() -> None:
    """build_substrate accepts the graph_type ExperimentConfig.default() ships ('lfr').

    The paper-scale n_nodes (~250k) is shrunk to a small, fast LFR; only the graph_type
    (which previously triggered a hard ValueError) is the property under test.
    """
    base = ExperimentConfig.default()
    assert base.graph.graph_type == "lfr"  # the shipped reference graph family

    small_lfr = replace(
        base.graph,
        n_nodes=400,
        lfr_min_community=15,
        lfr_max_community=60,
        lfr_max_degree=40,
        lfr_average_degree=6,
        seed=1,
    )
    cfg = replace(base, graph=small_lfr)
    mc = MonteCarloConfig(master_seed=7)

    with warnings.catch_warnings():
        # The default tau1=2.5 is heavy-tailed by design; that warning is orthogonal here.
        warnings.simplefilter("ignore")
        substrate = build_substrate(cfg, mc)

    assert isinstance(substrate, EgoSubstrate)
    assert substrate.num_nodes > 0
    assert substrate.k == cfg.model.k


def test_build_substrate_still_accepts_ba() -> None:
    """The BA branch (the shipped mc_default runner config) still builds (regression)."""
    cfg = ExperimentConfig.mc_default()
    assert cfg.graph.graph_type == "ba"
    cfg = replace(cfg, graph=replace(cfg.graph, n_nodes=300))
    mc = MonteCarloConfig(master_seed=7)
    substrate = build_substrate(cfg, mc)
    assert isinstance(substrate, EgoSubstrate)
    assert substrate.num_nodes > 0


def test_build_substrate_rejects_unknown_graph_type() -> None:
    """An unknown graph_type is rejected attributably (the dispatch is closed, not silent).

    GraphConfig validates graph_type at construction, so the rejection is forced through
    the build path by bypassing that validator via object.__setattr__ on the frozen config.
    """
    cfg = ExperimentConfig.mc_default()
    bad_graph = replace(cfg.graph, n_nodes=300)
    object.__setattr__(bad_graph, "graph_type", "watts_strogatz")
    cfg = replace(cfg, graph=bad_graph)
    mc = MonteCarloConfig(master_seed=7)
    with pytest.raises(ValueError, match="graph_type must be"):
        build_substrate(cfg, mc)
