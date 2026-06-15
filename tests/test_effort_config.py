"""Regression tests for effort-game neighborhood/depth consistency and the shipped
configs' instance-noise anneal-to-zero invariant (IZ Sec 4.2)."""

from __future__ import annotations

import pytest
import torch

from adversarial_networks import RootedMPNNDiscriminator
from adversarial_networks.config import (
    DEFAULT_TAIL_WINDOW,
    EffortExperimentConfig,
    EffortModelConfig,
    ExperimentConfig,
    InstanceNoiseConfig,
    MonteCarloConfig,
    assert_blur_anneals_to_zero_by_tail_window,
)
from adversarial_networks.core.objective import instance_noise_taus
from adversarial_networks.estimator_config import EstimatorConfig


def test_effort_defaults_use_2_hops_with_matched_sampling_and_depth() -> None:
    """Default effort config should target 2-hop neighborhoods with matched depth.

    Assertions track the shipped ``EffortExperimentConfig.default()`` values.
    """
    cfg = EffortExperimentConfig.default()
    assert cfg.model.k == 2
    assert cfg.model.resolved_discriminator_layers() == 2
    assert cfg.training.n_steps == 2000
    assert cfg.training.n_disc == 1
    assert cfg.training.lr_d == 2e-4
    assert cfg.training.lr_g == 7e-3
    assert cfg.training.grad_clip_norm_g == 25.0
    assert cfg.training.resolved_root_sampler_mode() == "disjoint_best_of_k"
    assert cfg.training.root_exclusion_r == 4


def test_effort_config_rejects_discriminator_depth_below_k() -> None:
    """Discriminator depth must still cover the chosen ego radius."""
    cfg = EffortExperimentConfig.default()
    with pytest.raises(ValueError, match="resolved discriminator_layers must be >= k"):
        EffortExperimentConfig(
            graph=cfg.graph,
            model=EffortModelConfig(k=2, discriminator_layers=1),
            training=cfg.training,
            instance_noise=cfg.instance_noise,
            true_params=cfg.true_params,
            init_params=cfg.init_params,
        )


@pytest.mark.parametrize(
    "factory",
    [
        ExperimentConfig.default,
        ExperimentConfig.mc_default,
        EffortExperimentConfig.default,
    ],
)
def test_shipped_blur_configs_anneal_to_zero_by_horizon(factory) -> None:
    """Every shipped blur-enabled config must reach exactly zero blur by its own
    training horizon, with anneal_steps<=n_steps and min_tau==0 (IZ Sec 4.2 s_anneal in
    {1..N_steps}, sigma->0). This pins the D10-02 invariant at the preset boundary."""
    cfg = factory()
    if not cfg.instance_noise.enabled:
        pytest.skip("blur disabled")
    n_steps = cfg.training.n_steps
    assert instance_noise_taus(cfg.instance_noise, generator_step=n_steps) == 0.0
    assert cfg.instance_noise.anneal_steps <= n_steps
    assert cfg.instance_noise.min_tau == 0.0


def test_experiment_config_rejects_residual_blur_at_construction() -> None:
    """ExperimentConfig.__post_init__ turns the runtime-only residual-blur warning into a
    hard config-construction ValueError when the blur does not reach zero by n_steps
    (D10-02)."""
    base = ExperimentConfig.default()
    with pytest.raises(ValueError, match="does not reach zero"):
        ExperimentConfig(
            graph=base.graph,
            model=base.model,
            training=base.training,  # n_steps=800
            instance_noise=InstanceNoiseConfig(
                enabled=True, tau_y0=1.0, schedule="linear",
                anneal_steps=10 * base.training.n_steps, min_tau=0.0,
            ),
            true_params=base.true_params,
            init_params=base.init_params,
        )


def test_effort_experiment_config_rejects_residual_blur_at_construction() -> None:
    """EffortExperimentConfig.__post_init__ also enforces anneal-to-zero (D10-02)."""
    base = EffortExperimentConfig.default()
    with pytest.raises(ValueError, match="does not reach zero"):
        EffortExperimentConfig(
            graph=base.graph,
            model=base.model,
            training=base.training,
            instance_noise=InstanceNoiseConfig(
                enabled=True, tau_y0=1.0, schedule="constant", anneal_steps=0, min_tau=0.0,
            ),
            true_params=base.true_params,
            init_params=base.init_params,
        )


def test_experiment_config_accepts_exp_blur_annealing_within_n_steps() -> None:
    """An 'exp' blur whose anneal_steps <= n_steps now constructs: the exp branch snaps to
    exactly 0.0 at step >= anneal_steps (D3-REG-exp-config-guard-strict-zero), so the strict
    _assert_blur_anneals_to_zero residual@n_steps check passes. Previously the continuous exp
    decay left a ~1e-35 residual and the guard hard-raised."""
    base = ExperimentConfig.default()
    cfg = ExperimentConfig(
        graph=base.graph,
        model=base.model,
        training=base.training,  # n_steps=800
        instance_noise=InstanceNoiseConfig(
            enabled=True, tau_y0=1.0, schedule="exp",
            anneal_steps=base.training.n_steps // 2, min_tau=0.0,
        ),
        true_params=base.true_params,
        init_params=base.init_params,
    )
    assert instance_noise_taus(cfg.instance_noise, generator_step=cfg.training.n_steps) == 0.0


def test_experiment_config_still_rejects_exp_blur_reaching_past_n_steps() -> None:
    """An 'exp' blur whose anneal_steps EXCEEDS n_steps still leaves residual blur at the
    horizon (the snap has not fired yet) and must hard-raise, confirming the guard is not
    blanket-disabled for exp."""
    base = ExperimentConfig.default()
    with pytest.raises(ValueError, match="does not reach zero"):
        ExperimentConfig(
            graph=base.graph,
            model=base.model,
            training=base.training,  # n_steps=800
            instance_noise=InstanceNoiseConfig(
                enabled=True, tau_y0=1.0, schedule="exp",
                anneal_steps=10 * base.training.n_steps, min_tau=0.0,
            ),
            true_params=base.true_params,
            init_params=base.init_params,
        )


def test_default_experiment_config_anneal_steps_clears_tail_window() -> None:
    """ExperimentConfig.default() sets anneal_steps to n_steps - DEFAULT_TAIL_WINDOW so the
    blur reaches zero BEFORE the tail-averaging window (not merely by the terminal step): the
    dataclass default anneal_steps=2000 would otherwise contaminate the tail window
    (D1-REG-container-blur-guard-terminal-vs-tailwindow)."""
    cfg = ExperimentConfig.default()
    assert cfg.training.n_steps == 800
    assert cfg.instance_noise.anneal_steps == cfg.training.n_steps - DEFAULT_TAIL_WINDOW == 700
    assert cfg.instance_noise.min_tau == 0.0
    # blur is exactly zero at the start of the tail-averaging window
    tail_start = cfg.training.n_steps - DEFAULT_TAIL_WINDOW + 1
    assert instance_noise_taus(cfg.instance_noise, generator_step=tail_start) == 0.0


def test_from_configs_validates_blur_against_resolved_runner_horizon() -> None:
    """EstimatorConfig.from_configs validates the blur against the RESOLVED runner horizon
    (MonteCarloConfig.max_steps), not only the standalone training.n_steps the container
    guard sees: a config whose blur clears the container guard at its n_steps horizon still
    raises when the runner max_steps is small enough to leave residual blur in the (shorter)
    tail-averaging window (D8-REG-blur-guard-wrong-horizon)."""
    from dataclasses import replace

    base = ExperimentConfig.mc_default()
    # anneal_steps=1000 <= n_steps - DEFAULT_TAIL_WINDOW (2000-100=1900) -> the blur reaches
    # zero before the n_steps=2000 tail window, so the container __post_init__ guard passes.
    cfg = replace(
        base,
        training=replace(base.training, n_steps=2000),
        instance_noise=InstanceNoiseConfig(
            enabled=True, tau_y0=1.0, schedule="linear", anneal_steps=1000, min_tau=0.0
        ),
    )
    # ... but the runner horizon is only 300 steps: tail_window=max(100,30)=100 -> the tail
    # window starts at step 201, where the (anneal_steps=1000) blur is still ~0.8 > 0.
    mc_short = MonteCarloConfig(max_steps=300, min_steps=0, convergence_window=100, stability_window=30)
    with pytest.raises(ValueError, match="tail-averaging window"):
        EstimatorConfig.from_configs(cfg, mc_short)

    # A blur that anneals before the resolved tail window (anneal_steps=100 <= 300-100) is
    # accepted at the same short runner horizon.
    cfg_ok = replace(
        base,
        training=replace(base.training, n_steps=2000),
        instance_noise=InstanceNoiseConfig(
            enabled=True, tau_y0=1.0, schedule="linear", anneal_steps=100, min_tau=0.0
        ),
    )
    ec = EstimatorConfig.from_configs(cfg_ok, mc_short)
    assert ec.max_steps == 300


def test_from_configs_accepts_shipped_mc_default_at_runner_horizon() -> None:
    """The shipped mc_default blur (anneal_steps=800) is valid at the shipped runner horizon
    (MonteCarloConfig default max_steps=2000): from_configs must NOT raise for it."""
    cfg = ExperimentConfig.mc_default()
    mc = MonteCarloConfig()  # max_steps=2000, tail window starts at 1901; blur is 0 by 800.
    ec = EstimatorConfig.from_configs(cfg, mc)
    assert ec.max_steps == 2000


def _container_accepts(cfg, blur: InstanceNoiseConfig) -> bool:
    """The container ``__post_init__`` blur verdict: True iff rebuilding ``cfg`` with ``blur``
    constructs (the guard accepts), False iff ``__post_init__`` raises (the guard rejects)."""
    from dataclasses import replace

    try:
        replace(cfg, instance_noise=blur)
    except ValueError:
        return False
    return True


def _from_configs_accepts(blur: InstanceNoiseConfig, n_steps: int) -> bool:
    """The from_configs blur verdict at the resolved-runner horizon ``max_steps=None`` (which
    resolves to ``n_steps``) with the default MonteCarloConfig windows. True iff the
    tail-window predicate from_configs invokes accepts, False iff it rejects.

    This calls the SAME ``assert_blur_anneals_to_zero_by_tail_window`` from_configs runs, at the
    SAME horizon (``max_steps=None`` -> ``n_steps``) and the SAME ``tail_window`` (default
    ``convergence_window=100``, ``stability_window=30`` -> ``max=100 == DEFAULT_TAIL_WINDOW``),
    so a contaminated config that ``__post_init__`` cannot even construct is still exercised
    against from_configs' predicate at that horizon.
    """
    mc = MonteCarloConfig(max_steps=None, convergence_window=100, stability_window=30)
    tail_window = max(int(mc.convergence_window), int(mc.stability_window))
    assert tail_window == DEFAULT_TAIL_WINDOW
    try:
        assert_blur_anneals_to_zero_by_tail_window(
            blur, n_steps, tail_window, "from_configs-agreement-probe"
        )
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    "factory",
    [
        ExperimentConfig.default,
        ExperimentConfig.mc_default,
        EffortExperimentConfig.default,
    ],
)
def test_container_and_from_configs_guards_agree_at_own_horizon(factory) -> None:
    """The container ``__post_init__`` guard and ``EstimatorConfig.from_configs`` return the
    SAME blur verdict on a config run at its own n_steps horizon (max_steps=None -> n_steps),
    closing the terminal-vs-tail-window contradiction
    (D1-REG-container-blur-guard-terminal-vs-tailwindow).

    (i) Each shipped factory at its own horizon: BOTH accept.
    (ii) A deliberately contaminated blur (anneal_steps = n_steps, so the blur is still
         positive at the tail-window start): BOTH reject.
    """
    cfg = factory()
    n_steps = cfg.training.n_steps
    shipped_blur = cfg.instance_noise

    # (i) shipped factory blur at its own horizon -> BOTH accept (same verdict).
    container_ok = _container_accepts(cfg, shipped_blur)
    from_configs_ok = _from_configs_accepts(shipped_blur, n_steps)
    assert container_ok is from_configs_ok is True
    # the real from_configs call (max_steps=None -> resolves to n_steps) also succeeds.
    mc_own = MonteCarloConfig(max_steps=None)
    ec = EstimatorConfig.from_configs(cfg, mc_own)
    assert ec.max_steps == n_steps

    # (ii) contaminated blur (anneal_steps == n_steps) -> BOTH reject (same verdict).
    contaminated = InstanceNoiseConfig(
        enabled=True, tau_y0=1.0, schedule="linear", anneal_steps=n_steps, min_tau=0.0
    )
    container_bad = _container_accepts(cfg, contaminated)
    from_configs_bad = _from_configs_accepts(contaminated, n_steps)
    assert container_bad is from_configs_bad is False


def test_instance_noise_rejects_floor_above_start() -> None:
    """InstanceNoiseConfig rejects min_tau > tau_y0 (a floor above the start would raise
    the blur above tau_y0; the decay schedule is non-increasing) (D1-05R)."""
    with pytest.raises(ValueError, match="min_tau"):
        InstanceNoiseConfig(tau_y0=0.05, min_tau=0.5)


def test_discriminator_forward_supports_2_message_passing_layers() -> None:
    """Variable discriminator depth should run a valid forward pass."""
    disc = RootedMPNNDiscriminator(hidden_dim=16, num_layers=2, logit_clip=10.0)
    x = torch.randn(5, 3, dtype=torch.float32)
    x[0, 2] = 1.0
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    root_indices = torch.tensor([0], dtype=torch.long)
    logits = disc(x, edge_index, root_indices)
    assert logits.shape == (1,)
