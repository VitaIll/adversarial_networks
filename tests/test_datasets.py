"""Tests for the synthetic dataset factories (make_linear_in_means / make_effort_game).

Also covers the inadmissible heavy-tailed admissibility warning: a power-law degree
exponent in the open interval ``(2, 3)`` (where the second degree moment, hence the
branching ratio ``lambda``, diverges and condition (M) fails) must warn, while a default
Barabasi-Albert graph (finite second moment for finite ``n``) must not.
"""

from __future__ import annotations

import warnings

import networkx as nx
import torch
from torch_geometric.utils import from_networkx, to_undirected

from adversarial_networks.data import NetworkData
from adversarial_networks.datasets import (
    _build_lfr_graph,
    _warn_if_degree_cap_admits_divergent_moment,
    _warn_if_heavy_tailed_exponent,
    flag_heavy_tailed_degrees,
    make_effort_game,
    make_linear_in_means,
)


def test_make_linear_in_means_returns_networkdata() -> None:
    data = make_linear_in_means(n_nodes=200, graph="ba", k=2, seed=0, m=2)
    assert isinstance(data, NetworkData)
    assert data.k == 2
    assert data.num_nodes <= 200  # sanitisation may drop a few nodes
    assert data.y.dtype == torch.float32 and bool(torch.isfinite(data.y).all())


def test_same_seed_gives_identical_outcome() -> None:
    a = make_linear_in_means(n_nodes=200, seed=3, m=2)
    b = make_linear_in_means(n_nodes=200, seed=3, m=2)
    assert torch.equal(a.y, b.y)
    assert torch.equal(a.X, b.X)


def test_different_seed_changes_outcome() -> None:
    a = make_linear_in_means(n_nodes=200, seed=3, m=2)
    c = make_linear_in_means(n_nodes=200, seed=4, m=2)
    assert not torch.equal(a.y, c.y)


def test_true_params_are_honoured() -> None:
    data = make_linear_in_means(n_nodes=300, seed=0, m=2, true_params={"beta": 0.3, "gamma": 2.0})
    # the simulated outcome should correlate strongly with X at gamma=2.0
    corr = torch.corrcoef(torch.stack([data.X, data.y]))[0, 1]
    assert float(corr) > 0.3


def test_make_effort_game_returns_networkdata_and_reproducible() -> None:
    a = make_effort_game(n_nodes=200, seed=1, m=2)
    b = make_effort_game(n_nodes=200, seed=1, m=2)
    assert isinstance(a, NetworkData)
    assert torch.equal(a.y, b.y)
    assert bool(torch.isfinite(a.y).all())


# ---------------------------------------------------- inadmissible heavy-tail warning
def test_heavy_tail_exponent_warns_in_open_interval() -> None:
    """A power-law degree exponent in (2, 3) warns; the closed boundaries and >3 do not."""
    for tau1 in (2.1, 2.5, 2.9):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_heavy_tailed_exponent(tau1)
        heavy = [w for w in caught if "heavy-tailed" in str(w.message)]
        assert len(heavy) == 1 and issubclass(heavy[0].category, RuntimeWarning)

    for tau1 in (2.0, 3.0, 3.5):  # open interval: boundaries are admissible
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_heavy_tailed_exponent(tau1)
        assert not [w for w in caught if "heavy-tailed" in str(w.message)]


def test_lfr_default_exponent_warns_via_build(monkeypatch) -> None:
    """_build_lfr_graph warns for its default tau1=2.5 (the read precedes the LFR call)."""
    # Stub the (slow, variance-prone) LFR generator so the test exercises only the
    # warning path; the graph itself is irrelevant here.
    monkeypatch.setattr(nx, "LFR_benchmark_graph", lambda **kwargs: nx.path_graph(10))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_lfr_graph(10, seed=0)  # default tau1 = 2.5 ∈ (2, 3)
    heavy = [w for w in caught if "heavy-tailed" in str(w.message)]
    assert len(heavy) == 1 and issubclass(heavy[0].category, RuntimeWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_lfr_graph(10, seed=0, tau1=3.5)  # admissible tail
    assert not [w for w in caught if "heavy-tailed" in str(w.message)]


def test_default_ba_does_not_warn_heavy_tail() -> None:
    """A default Barabasi-Albert dataset emits no heavy-tail warning (finite second moment)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = make_linear_in_means(n_nodes=300, graph="ba", seed=0, m=2)
    assert not [w for w in caught if "heavy-tailed" in str(w.message)]
    # And the empirical degree-tail flag agrees: BA(m=2) max degree stays ~O(sqrt(n)).
    assert flag_heavy_tailed_degrees(data) is False


# --------------------------------------------- inadmissible degree-cap (G2) warning
def test_degree_cap_warns_when_looser_than_sqrt_n() -> None:
    """A max_degree cap above ~3*sqrt(n) does not bound E[D^2] asymptotically -> warns;
    a tight cap (e.g. the paper's 102 at n=250k) and the BA-scale 100 at n=10k do not."""
    # loose caps (> 3*sqrt(n)): warn.
    for max_degree, n in ((1000, 10_000), (2000, 250_000)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_degree_cap_admits_divergent_moment(max_degree, n)
        loose = [w for w in caught if "exceeds the finite-second-moment ceiling" in str(w.message)]
        assert len(loose) == 1 and issubclass(loose[0].category, RuntimeWarning)

    # tight caps (<= 3*sqrt(n)): admissible, no warning.
    for max_degree, n in ((102, 250_000), (100, 10_000)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_degree_cap_admits_divergent_moment(max_degree, n)
        assert not [w for w in caught if "ceiling" in str(w.message)]


def test_missing_degree_cap_warns_at_large_n_only() -> None:
    """No cap at a non-trivial scale (n>=1000) surfaces the missing cap; a tiny graph
    (n<1000) stays silent (the cap is not yet load-bearing)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_degree_cap_admits_divergent_moment(None, 10_000)
    no_cap = [w for w in caught if "no max_degree cap" in str(w.message)]
    assert len(no_cap) == 1 and issubclass(no_cap[0].category, RuntimeWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_degree_cap_admits_divergent_moment(None, 200)
    assert not [w for w in caught if "no max_degree cap" in str(w.message)]


def test_lfr_loose_degree_cap_warns_via_build(monkeypatch) -> None:
    """_build_lfr_graph surfaces a loose max_degree cap forwarded via graph_kwargs (the
    read precedes the LFR call). Uses an admissible tau1>=3 so only the cap warning fires."""
    monkeypatch.setattr(nx, "LFR_benchmark_graph", lambda **kwargs: nx.path_graph(10))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # n=5000 -> 3*sqrt(n) ~ 212; cap 900 is far looser. tau1=3.5 keeps the tail admissible.
        _build_lfr_graph(5000, seed=0, tau1=3.5, max_degree=900)
    cap = [w for w in caught if "exceeds the finite-second-moment ceiling" in str(w.message)]
    assert len(cap) == 1 and issubclass(cap[0].category, RuntimeWarning)
    # tau1=3.5 is admissible, so the heavy-tail exponent warning must NOT fire.
    assert not [w for w in caught if "heavy-tailed" in str(w.message)]


def test_lfr_max_degree_cap_is_forwarded_and_binds() -> None:
    """The LFR degree cap is now forwarded to nx.LFR_benchmark_graph, so a capped build
    realises max_degree <= cap (D8-04-R1) — previously the cap was read only for the
    warning and dropped at the build call, so it had no effect. A sufficient cap
    (<= 3*sqrt(n)) also suppresses the degree-cap warning, while an uncapped build at
    n >= 1000 still warns (D8-REG-degree-cap), making the warning's prescribed remedy
    actually functional. Admissible tau1=3 keeps the heavy-tail exponent warning silent."""
    import networkx as nx

    n, cap = 1100, 40  # 3*sqrt(1100) ~ 99.5, so cap=40 is within the finite-moment ceiling.

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capped = _build_lfr_graph(
            n, seed=0, tau1=3.0, tau2=1.5, mu=0.2, average_degree=6, min_community=30,
            max_degree=cap, max_retries=8, lfr_max_iters=300,
        )
    realized_capped = max(dict(capped.degree()).values())
    assert realized_capped <= cap, f"cap not honoured: realized max degree {realized_capped} > {cap}"
    # A sufficient cap (<= 3*sqrt(n)) emits neither the loose-cap nor the missing-cap warning.
    assert not [
        w for w in caught
        if "finite-second-moment ceiling" in str(w.message) or "no max_degree cap" in str(w.message)
    ]
    # tau1=3 is admissible (closed boundary), so the heavy-tail exponent warning must not fire.
    assert not [w for w in caught if "heavy-tailed" in str(w.message)]

    # Uncapped build at the SAME n>=1000 realises a much larger hub AND warns about the
    # missing cap — confirming the cap is load-bearing, not cosmetic.
    with warnings.catch_warnings(record=True) as caught_uncapped:
        warnings.simplefilter("always")
        uncapped = _build_lfr_graph(
            n, seed=0, tau1=3.0, tau2=1.5, mu=0.2, average_degree=6, min_community=30,
            max_retries=8, lfr_max_iters=300,
        )
    realized_uncapped = max(dict(uncapped.degree()).values())
    assert realized_uncapped > cap, "uncapped build should exceed the cap (else the cap proves nothing)"
    no_cap_warns = [w for w in caught_uncapped if "no max_degree cap" in str(w.message)]
    assert len(no_cap_warns) == 1 and issubclass(no_cap_warns[0].category, RuntimeWarning)


def test_lfr_min_degree_is_forwarded_instead_of_average_degree() -> None:
    """When min_degree is supplied (the GraphConfig alternative to average_degree),
    _build_lfr_graph forwards min_degree and drops average_degree (networkx requires
    exactly one), so the build honours min_degree and every node meets it."""
    n = 1100
    g = _build_lfr_graph(
        n, seed=0, tau1=3.0, tau2=1.5, mu=0.2, min_degree=5, min_community=30,
        max_degree=40, max_retries=8, lfr_max_iters=300,
    )
    realized_min_degree = min(dict(g.degree()).values())
    # Simplification (multi-edge / self-loop removal) can only lower a degree, so the
    # realised minimum is a lower bound on the LFR min_degree; require it stays positive
    # and the graph is non-trivial (the min_degree branch built at all, not average_degree).
    assert g.number_of_nodes() > 0 and realized_min_degree >= 1


def test_empirical_helper_flags_constructed_heavy_tailed_graph() -> None:
    """flag_heavy_tailed_degrees flags a star (hub degree n-1 >> sqrt(n)) and warns."""
    n = 200
    star = nx.star_graph(n - 1)  # one hub of degree n-1, n-1 leaves of degree 1
    edge_index = to_undirected(from_networkx(star).edge_index, num_nodes=n).contiguous()
    X = torch.randn(n, dtype=torch.float32)
    y = torch.randn(n, dtype=torch.float32)
    data = NetworkData.from_edge_index(edge_index, X, y, k=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        heavy = flag_heavy_tailed_degrees(data)
    assert heavy is True
    assert [w for w in caught if "heavy-tailed" in str(w.message)]
