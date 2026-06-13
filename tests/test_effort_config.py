"""Regression tests for effort-game neighborhood/depth consistency."""

from __future__ import annotations

import pytest
import torch

from src import EffortExperimentConfig, EffortModelConfig, RootedMPNNDiscriminator


def test_effort_defaults_use_2_hops_with_matched_sampling_and_depth() -> None:
    """Default effort config should target 2-hop neighborhoods with matched depth."""
    cfg = EffortExperimentConfig.default()
    assert cfg.model.k == 2
    assert cfg.model.resolved_discriminator_layers() == 2
    assert cfg.training.n_steps == 2000
    assert cfg.training.n_disc == 1
    assert cfg.training.lr_d == 2e-4
    assert cfg.training.lr_g == 3e-3
    assert cfg.training.grad_clip_norm_g == 20.0
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
