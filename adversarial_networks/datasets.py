"""Synthetic ``NetworkData`` factories (sklearn ``make_*`` idiom).

Each factory generates a graph + covariates, simulates the *observed* equilibrium
from the corresponding built-in model at known ``true_params``, and returns a single
:class:`~adversarial_networks.data.NetworkData`. The true parameters are an
*input* (not part of a polymorphic return — no ``return_X`` variants).

One documented RNG lineage makes the data reproducible: the **graph** uses ``seed``,
the **covariates** ``seed + 1``, and the **outcome** ``seed + 2``. The same ``seed``
therefore yields an identical ``y``.
"""

from __future__ import annotations

import math
import warnings

import networkx as nx
import torch

from .data import NetworkData
from .ego import EgoSubstrate
from .generators import EffortGameGenerator, LinearInMeansGenerator

_LINEAR_TRUE = {"beta": 0.4, "gamma": 1.5, "sigma_sq": 1.0}
_EFFORT_TRUE = {"gamma": 1.5, "lambda_": 2.0 / 3.0, "mu": 0.5, "r": 1.0, "sigma_sq": 1.0}


def _warn_if_heavy_tailed_exponent(tau1: float) -> None:
    """Warn if a power-law degree exponent lies in the inadmissible regime ``(2, 3)``.

    A degree tail index ``tau1 in (2, 3)`` gives a divergent second degree moment, hence
    an infinite branching ratio ``lambda`` (Illichmann & Zacchia 2026, finite-moment note,
    Primitive 2.7): condition (M) ``rho_bar^p lambda < 1`` and the volume-growth bound (G2)
    both fail, and the consistency rate degrades. The boundary values ``2`` and ``3`` are
    *not* flagged (the interval is open).
    """
    if 2.0 < float(tau1) < 3.0:
        warnings.warn(
            f"Power-law degree exponent tau1={float(tau1):g} is in the inadmissible "
            "heavy-tailed regime (2, 3): the second degree moment diverges, so the branching "
            "ratio lambda is infinite and the finite-moment condition (M) rho_bar^p*lambda<1 "
            "(and the volume-growth bound G2) fail — the consistency rate degrades. Use "
            "tau1 >= 3 for a finite-moment (admissible) degree tail.",
            RuntimeWarning,
            stacklevel=3,
        )


def _warn_if_degree_cap_admits_divergent_moment(
    max_degree: int | None,
    n_nodes: int,
    *,
    factor: float = 3.0,
) -> None:
    """Warn if the LFR degree *cap* is too loose to enforce a finite second degree moment.

    The degree cap ``max_degree`` is the mechanism that keeps ``E[D^2]`` (hence the
    branching ratio ``lambda = E[D(D-1)]/E[D]``) finite when the degree law has a heavy tail
    — it is the only LFR knob that can rescue an admissible (G2 / finite-moment) ensemble
    from a divergent second moment (Illichmann & Zacchia 2026, finite-moment note, Primitive
    2.7; cf. the paper's capped LFR at ``n~250k`` with max degree ``102``). For the
    *ensemble* (``n -> inf``) to stay admissible the cap must grow no faster than ``sqrt(n)``
    (a finite second moment caps the max degree at ``O(sqrt(n))``); a cap scaling faster than
    ``sqrt(n)`` does not actually bound ``E[D^2]`` asymptotically and permits hubs whose
    presence diverges the second moment.

    Two configurations are surfaced (observability only — the graph is built unchanged):

    * **A loose cap**: ``max_degree`` is set but exceeds ``factor * sqrt(n)`` — the cap is too
      high to enforce the finite-second-moment ceiling.
    * **No cap on a large graph**: ``max_degree is None`` at ``n`` large enough that the
      uncapped LFR degree law can realise hubs above the ``sqrt(n)`` ceiling; without a cap
      the second moment is governed solely by the tail exponent (see
      :func:`_warn_if_heavy_tailed_exponent`).

    This is distinct from the ``tau1``-exponent warning (which flags the *degree-law knob*);
    this flags the *degree-cap knob*, so the two do not double-warn the same parameter.
    """
    n = int(n_nodes)
    if n <= 0:
        return
    ceiling = float(factor) * float(n) ** 0.5
    if max_degree is not None:
        if int(max_degree) > ceiling:
            warnings.warn(
                f"LFR max_degree cap ({int(max_degree)}) exceeds the finite-second-moment "
                f"ceiling ~ {factor:g}*sqrt(n) = {ceiling:.1f} (n={n}): a degree cap scaling "
                "faster than sqrt(n) does not bound E[D^2] as n grows, so the branching ratio "
                "lambda can diverge and the finite-moment condition (M)/G2 fail. Use a cap "
                "that grows no faster than sqrt(n) (the paper caps an n~250k LFR graph at max "
                "degree 102) for an admissible ensemble.",
                RuntimeWarning,
                stacklevel=3,
            )
    elif ceiling >= 1.0 and n >= 1000:
        # No cap at a non-trivial scale: the second moment is then governed solely by the
        # tail exponent, with nothing to rescue a heavy tail. Surface the missing cap.
        warnings.warn(
            f"LFR built with no max_degree cap at n={n}: without a degree cap the second "
            f"degree moment is governed solely by the tail exponent and the uncapped law can "
            f"realise hubs above the finite-second-moment ceiling ~ sqrt(n) = {n ** 0.5:.1f}. "
            "If the degree tail is heavy (tau1 in (2, 3)), the branching ratio lambda diverges "
            "(condition (M)/G2 fail); set lfr_max_degree to a cap growing no faster than "
            "sqrt(n) to keep the ensemble admissible.",
            RuntimeWarning,
            stacklevel=3,
        )


def flag_heavy_tailed_degrees(network, *, exponent: float = 0.5, factor: float = 3.0) -> bool:
    """Post-hoc check: is a built graph's degree distribution empirically heavy-tailed?

    A complement to :func:`_warn_if_heavy_tailed_exponent` for graphs whose tail exponent
    is not exposed as a parameter. In the inadmissible regime the maximum degree grows
    faster than ``sqrt(n)`` (a finite second degree moment caps the max degree at
    ``O(sqrt(n))``; a tail index in ``(2, 3)`` lets it grow like ``n^{1/(tau1-1)}`` with
    ``1/(tau1-1) > 1/2``). This flags ``max_degree > factor * n**exponent`` and emits a
    ``RuntimeWarning`` when it fires; the graph is not modified.

    Args:
        network: An object exposing ``.W`` (the coalesced sparse interaction matrix whose
            indices are the graph adjacency), e.g. an ``EgoSubstrate`` / ``NetworkData``.
        exponent: Growth exponent threshold for ``max_degree`` vs ``n`` (``0.5`` ⇒ the
            finite-second-moment ceiling ``sqrt(n)``).
        factor: Multiplicative slack on the ``n**exponent`` threshold.

    Returns:
        ``True`` iff the empirical max degree exceeds ``factor * n**exponent``.
    """
    from .generators import _extract_w_x  # local import: avoid a module import cycle

    W, _ = _extract_w_x(network)
    indices = W.coalesce().indices()
    num_nodes = int(W.shape[0])
    max_degree = int(torch.bincount(indices[0], minlength=num_nodes).max().item())
    threshold = float(factor) * float(num_nodes) ** float(exponent)
    heavy = max_degree > threshold
    if heavy:
        warnings.warn(
            f"Empirically heavy-tailed degrees: max_degree={max_degree} exceeds "
            f"{factor:g}*n^{exponent:g}={threshold:.1f} (n={num_nodes}). The max degree "
            "grows faster than sqrt(n), consistent with a divergent second degree moment "
            "(infinite branching lambda) — the finite-moment condition (M) and rate are at "
            "risk.",
            RuntimeWarning,
            stacklevel=2,
        )
    return heavy


def _build_lfr_graph(n_nodes: int, seed: int, **kwargs) -> nx.Graph:
    """Build an LFR community graph, retrying across seeds (LFR can fail to converge).

    ``tau1`` is the power-law exponent of the degree distribution. A value in the open
    interval ``(2, 3)`` is the *inadmissible heavy-tailed regime*: the second degree
    moment (hence the branching ratio ``lambda``) diverges, breaking (G2) and the
    finite-moment condition (M) and degrading the consistency rate (Illichmann & Zacchia
    2026, finite-moment note, Primitive 2.7 "Excluded"). Independently, the degree *cap*
    ``max_degree`` (passed via ``graph_kwargs``) is the knob that keeps ``E[D^2]`` finite
    under a heavy tail; a cap scaling faster than ``sqrt(n)`` (or no cap at a large ``n``)
    is surfaced by :func:`_warn_if_degree_cap_admits_divergent_moment`. Both are
    observability only — the graph is built unchanged; ``RuntimeWarning`` s flag the regime.
    """
    tau1 = float(kwargs.get("tau1", 2.5))
    _warn_if_heavy_tailed_exponent(tau1)
    max_degree_cap = kwargs.get("max_degree")
    _warn_if_degree_cap_admits_divergent_moment(
        None if max_degree_cap is None else int(max_degree_cap), n_nodes
    )
    tau2 = float(kwargs.get("tau2", 1.5))
    mu = float(kwargs.get("mu", 0.3))
    min_community = int(kwargs.get("min_community", 20))
    max_iters = int(kwargs.get("lfr_max_iters", 500))
    max_retries = int(kwargs.get("max_retries", 20))

    # Degree controls: networkx requires EXACTLY ONE of average_degree / min_degree. Honour
    # an explicit min_degree (the GraphConfig alternative), else default to average_degree=6.
    min_degree_cap = kwargs.get("min_degree")
    average_degree = kwargs.get("average_degree")
    if min_degree_cap is not None:
        average_degree = None
    elif average_degree is None:
        average_degree = 6
    max_community_cap = kwargs.get("max_community")

    # Forward the cap-related bounds (max_degree / max_community / min_degree) so a capped
    # LFR ensemble is actually buildable — these are the knobs that bound E[D^2] (hence the
    # branching ratio lambda) under a heavy tail (finite-moment Primitive 2.7); previously
    # they were read only for the warning and dropped at the build call, making the
    # warning's prescribed remedy inert (D8-04-R1 / D8-REG-degree-cap). networkx 3.6.1
    # accepts all three; pass only the ones provided (None means "unset").
    optional_bounds: dict[str, int] = {}
    if average_degree is not None:
        optional_bounds["average_degree"] = int(average_degree)
    if min_degree_cap is not None:
        optional_bounds["min_degree"] = int(min_degree_cap)
    if max_degree_cap is not None:
        optional_bounds["max_degree"] = int(max_degree_cap)
    if max_community_cap is not None:
        optional_bounds["max_community"] = int(max_community_cap)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            graph = nx.LFR_benchmark_graph(
                n=n_nodes, tau1=tau1, tau2=tau2, mu=mu,
                min_community=min_community, seed=seed + attempt, max_iters=max_iters,
                **optional_bounds,
            )
            simple = nx.Graph(graph)  # drop multi-edges
            simple.remove_edges_from(nx.selfloop_edges(simple))
            if simple.number_of_edges() > 0:
                return simple
        except (nx.ExceededMaxIterations, nx.NetworkXError) as exc:  # pragma: no cover - LFR variance
            last_exc = exc
    raise RuntimeError(
        f"LFR graph generation failed after {max_retries} retries (n={n_nodes}); "
        f"last error: {last_exc}"
    )


def _generate_graph(graph: str, n_nodes: int, seed: int, **graph_kwargs) -> nx.Graph:
    if graph == "ba":
        m = int(graph_kwargs.get("m", 2))
        return nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)
    if graph == "lfr":
        return _build_lfr_graph(n_nodes, seed, **graph_kwargs)
    raise ValueError(f"graph must be 'ba' or 'lfr', got {graph!r}.")


def make_linear_in_means(
    *,
    n_nodes: int = 10_000,
    graph: str = "ba",
    k: int = 2,
    true_params: dict[str, float] | None = None,
    beta_cap: float = 0.85,
    picard_tol: float = 1e-6,
    picard_max: int = 100,
    root_sampler_mode: str = "uniform",
    seed: int = 0,
    **graph_kwargs,
) -> NetworkData:
    """Synthetic linear-in-means data: ``Y = beta*W*Y + gamma*X + eps``.

    Args:
        n_nodes: Target node count.
        graph: ``"ba"`` (Barabasi-Albert) or ``"lfr"`` (community structure).
        k: Ego radius.
        true_params: Data-generating ``{beta, gamma, sigma_sq}`` (defaults provided).
        beta_cap, picard_tol, picard_max: True-model solver controls.
        root_sampler_mode: Root sampler for the substrate.
        seed: Master seed (graph=``seed``, X=``seed+1``, outcome=``seed+2``).
        **graph_kwargs: Forwarded to the graph generator (e.g. ``m`` for BA).

    Returns:
        A :class:`NetworkData` whose ``y`` is the simulated equilibrium at ``true_params``.
    """
    tp = dict(_LINEAR_TRUE)
    if true_params:
        tp.update(true_params)
    graph_obj = _generate_graph(graph, n_nodes, seed, **graph_kwargs)
    torch.manual_seed(seed + 1)
    X = torch.randn(graph_obj.number_of_nodes(), dtype=torch.float32)
    substrate = EgoSubstrate.from_networkx(
        graph_obj, X, k=k, root_sampler_mode=root_sampler_mode, seed=seed
    )
    true_model = LinearInMeansGenerator(
        beta_cap=beta_cap, picard_tol=picard_tol, picard_max=picard_max,
        init_beta=float(tp["beta"]), init_gamma=float(tp["gamma"]),
        init_log_sigma_sq=math.log(float(tp["sigma_sq"])),
    )
    torch.manual_seed(seed + 2)
    with torch.no_grad():
        y = true_model(substrate.W, substrate.X).detach().to(torch.float32)
    return NetworkData(substrate, y)


def make_effort_game(
    *,
    n_nodes: int = 10_000,
    graph: str = "ba",
    k: int = 2,
    true_params: dict[str, float] | None = None,
    lambda_max: float = 4.0,
    fix_r: float | None = 1.0,
    fix_sigma_sq: float | None = 1.0,
    picard_tol: float = 1e-6,
    picard_max: int = 100,
    newton_tol: float = 1e-10,
    newton_max: int = 8,
    root_sampler_mode: str = "uniform",
    seed: int = 0,
    **graph_kwargs,
) -> NetworkData:
    """Synthetic nonlinear effort-game data (implicit FOC best response).

    ``fix_r``/``fix_sigma_sq`` pin ``r``/``sigma^2`` (the finite-moment Lemma-2 regime);
    when fixed, the corresponding ``true_params`` entry equals the fixed value.
    RNG lineage as in :func:`make_linear_in_means`.
    """
    tp = dict(_EFFORT_TRUE)
    if true_params:
        tp.update(true_params)
    if fix_r is not None:
        tp["r"] = float(fix_r)
    if fix_sigma_sq is not None:
        tp["sigma_sq"] = float(fix_sigma_sq)

    graph_obj = _generate_graph(graph, n_nodes, seed, **graph_kwargs)
    torch.manual_seed(seed + 1)
    X = torch.randn(graph_obj.number_of_nodes(), dtype=torch.float32)
    substrate = EgoSubstrate.from_networkx(
        graph_obj, X, k=k, root_sampler_mode=root_sampler_mode, seed=seed
    )
    init_log_sigma_sq = 0.0 if fix_sigma_sq is not None else math.log(float(tp["sigma_sq"]))
    true_model = EffortGameGenerator(
        lambda_max=lambda_max, picard_tol=picard_tol, picard_max=picard_max,
        newton_tol=newton_tol, newton_max=newton_max, fix_r=fix_r, fix_sigma_sq=fix_sigma_sq,
        init_gamma=float(tp["gamma"]), init_lambda=float(tp["lambda_"]), init_mu=float(tp["mu"]),
        init_r=float(tp["r"]), init_log_sigma_sq=init_log_sigma_sq,
    )
    torch.manual_seed(seed + 2)
    with torch.no_grad():
        y = true_model(substrate.W, substrate.X).detach().to(torch.float32)
    return NetworkData(substrate, y)
