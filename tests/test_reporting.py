"""Tests for recovery_table reporting."""

from __future__ import annotations

import warnings

import pandas as pd

import adversarial_networks as an
from adversarial_networks.discriminator import RootedMPNNDiscriminator
from adversarial_networks.estimator import AdversarialEstimator
from adversarial_networks.estimator_config import EstimatorConfig
from adversarial_networks.generators import LinearInMeansGenerator
from adversarial_networks.reporting import recovery_table


def _fitted():
    data = an.make_linear_in_means(n_nodes=150, seed=0, m=2)
    model = LinearInMeansGenerator(beta_cap=0.85)
    disc = RootedMPNNDiscriminator(hidden_dim=8, num_layers=2, logit_clip=10.0)
    cfg = EstimatorConfig(max_steps=10, batch_size=8, n_disc=1, lr_d=1e-3, lr_g=5e-3, seed=7,
                          convergence_window=100, stability_window=30)
    est = AdversarialEstimator(model, disc, config=cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data)
    return est


def test_recovery_table_columns_and_values() -> None:
    est = _fitted()
    true = {"beta": 0.4, "gamma": 1.5, "sigma_sq": 1.0}
    table = recovery_table(est, true)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["coef", "true", "abs_err", "path_sd"]
    assert list(table.index) == ["beta", "gamma", "sigma_sq"]
    for name in true:
        assert table.loc[name, "true"] == true[name]
        assert abs(table.loc[name, "abs_err"] - abs(table.loc[name, "coef"] - true[name])) < 1e-9
        assert table.loc[name, "path_sd"] >= 0.0


def test_estimator_recovery_table_matches_free_function() -> None:
    est = _fitted()
    true = {"beta": 0.4, "gamma": 1.5, "sigma_sq": 1.0}
    a = est.recovery_table(true)
    b = recovery_table(est, true)
    assert a.equals(b)
