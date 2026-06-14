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

import networkx as nx
import torch

from .data import NetworkData
from .ego import EgoSubstrate
from .generators import EffortGameGenerator, LinearInMeansGenerator

_LINEAR_TRUE = {"beta": 0.4, "gamma": 1.5, "sigma_sq": 1.0}
_EFFORT_TRUE = {"gamma": 1.5, "lambda_": 2.0 / 3.0, "mu": 0.5, "r": 1.0, "sigma_sq": 1.0}


def _build_lfr_graph(n_nodes: int, seed: int, **kwargs) -> nx.Graph:
    """Build an LFR community graph, retrying across seeds (LFR can fail to converge)."""
    tau1 = float(kwargs.get("tau1", 2.5))
    tau2 = float(kwargs.get("tau2", 1.5))
    mu = float(kwargs.get("mu", 0.3))
    average_degree = int(kwargs.get("average_degree", 6))
    min_community = int(kwargs.get("min_community", 20))
    max_iters = int(kwargs.get("lfr_max_iters", 500))
    max_retries = int(kwargs.get("max_retries", 20))

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            graph = nx.LFR_benchmark_graph(
                n=n_nodes, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree,
                min_community=min_community, seed=seed + attempt, max_iters=max_iters,
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
