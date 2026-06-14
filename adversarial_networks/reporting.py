"""Recovery reporting for the adversarial estimator.

:func:`recovery_table` is the convenience summary for a *synthetic-recovery* study:
it lines up the estimated coefficients against the known data-generating
parameters and reports the absolute error and the optimisation-path spread. It is
**not** an inferential table — ``path_sd`` is an optimisation-convergence
diagnostic, not a standard error (see ``AdversarialEstimator.estimates_``).
"""

from __future__ import annotations

from collections.abc import Mapping


def recovery_table(estimator, true_params: Mapping[str, float]):
    """Compare estimated parameters to the data-generating truth.

    Args:
        estimator: A fitted :class:`~adversarial_networks.estimator.AdversarialEstimator`.
        true_params: The data-generating parameter values, keyed by parameter name.

    Returns:
        A ``pandas.DataFrame`` indexed by parameter with columns ``coef`` (the
        tail-averaged estimate), ``true``, ``abs_err`` (``|coef - true|``), and
        ``path_sd`` (the optimisation-path std — **not** a standard error). Rows are
        ordered by the estimator's ``feature_names_``.
    """
    import pandas as pd

    estimates = estimator.estimates_
    rows: dict[str, dict[str, float]] = {}
    for name in estimator.feature_names_:
        coef = float(estimates.loc[name, "coef"])
        path_sd = float(estimates.loc[name, "path_sd"])
        if name in true_params:
            true = float(true_params[name])
            abs_err = abs(coef - true)
        else:
            true = float("nan")
            abs_err = float("nan")
        rows[name] = {"coef": coef, "true": true, "abs_err": abs_err, "path_sd": path_sd}

    frame = pd.DataFrame.from_dict(rows, orient="index", columns=["coef", "true", "abs_err", "path_sd"])
    frame.index.name = "param"
    return frame
