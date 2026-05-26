"""Pearson and Spearman correlation analysis between features and targets."""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

CorrelationMethod = Literal["pearson", "spearman", "both"]

_VALID_METHODS: frozenset[str] = frozenset({"pearson", "spearman", "both"})


def run_correlations(
    X: pd.DataFrame,
    y: pd.Series,
    method: CorrelationMethod = "both",
    significance_threshold: float = 0.05,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute feature-target correlations with significance testing.

    Continuous features are assessed with Pearson and/or Spearman
    correlations.  Categorical features (e.g. an encoded catalyst type)
    are assessed with **one-way ANOVA** and **eta-squared (η²)** — the
    correct statistical test when the predictor is unordered discrete
    groups and the target is continuous.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : pd.Series
        Target variable with length n_samples.
    method : {'pearson', 'spearman', 'both'}, optional
        Correlation method(s) to compute for continuous features.
        Default is ``'both'``.
    significance_threshold : float, optional
        p-value threshold below which a result is considered
        statistically significant. Default is 0.05.
    categorical_cols : list of str or None, optional
        Names of columns in *X* that are categorical (unordered discrete).
        For these features ANOVA + eta-squared is computed instead of
        Pearson/Spearman.  Pass ``list(cat_maps.keys())`` from
        :func:`~nextscreen.data.loader.encode_categoricals`.
        Default ``None`` (all features treated as continuous).

    Returns
    -------
    pd.DataFrame
        DataFrame **indexed by feature name**.

        Continuous features have columns (depending on *method*):

        - ``pearson_r``, ``pearson_p``, ``pearson_significant``
        - ``spearman_r``, ``spearman_p``, ``spearman_significant``

        Categorical features have columns:

        - ``eta_squared`` — ANOVA effect size (η², range [0, 1])
        - ``anova_p``     — ANOVA p-value
        - ``anova_significant``

        Columns that do not apply to a feature are ``NaN``.
        A ``'rank'`` column is always appended; categorical features are
        ranked by ``sqrt(eta_squared)`` (the correlation ratio η, range
        [0, 1]) so the ranking is directly comparable to ``|r|`` for
        continuous features.

    Raises
    ------
    ValueError
        If *method* is not one of the accepted literals, if *X* is empty,
        or if *X* and *y* have different lengths.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid options: {sorted(_VALID_METHODS)}"
        )
    if X.empty:
        raise ValueError("X must contain at least one row.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length "
            f"({len(X)} vs {len(y)})."
        )

    cat_set: frozenset[str] = frozenset(categorical_cols or [])
    y_np = y.to_numpy(dtype=float)

    nan = float("nan")
    records: list[dict[str, object]] = []
    for feature in X.columns:
        row: dict[str, object] = {}

        if feature in cat_set:
            # --- Categorical: one-way ANOVA + eta-squared ---
            # Use raw column values so string categories work correctly.
            col_raw = X[feature].to_numpy()
            # NaN placeholders for Pearson/Spearman columns
            if method in ("pearson", "both"):
                row["pearson_r"] = nan
                row["pearson_p"] = nan
                row["pearson_significant"] = nan
            if method in ("spearman", "both"):
                row["spearman_r"] = nan
                row["spearman_p"] = nan
                row["spearman_significant"] = nan

            groups = [
                y_np[col_raw == v]
                for v in np.unique(col_raw)
                if np.sum(col_raw == v) > 0
            ]
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                _, p_anova = stats.f_oneway(*groups)
                eta2 = _eta_squared(groups, y_np)
            else:
                p_anova = nan
                eta2 = nan

            row["eta_squared"] = eta2
            row["anova_p"] = float(p_anova) if not math.isnan(float(p_anova if p_anova is not None else nan)) else nan
            row["anova_significant"] = (
                bool(float(p_anova) < significance_threshold)
                if not math.isnan(float(p_anova if p_anova is not None else nan))
                else nan
            )
        else:
            # --- Continuous: Pearson and/or Spearman ---
            col = X[feature].to_numpy(dtype=float)
            if method in ("pearson", "both"):
                r_p, p_p = stats.pearsonr(col, y_np)
                row["pearson_r"] = float(r_p)
                row["pearson_p"] = float(p_p)
                row["pearson_significant"] = bool(
                    p_p < significance_threshold
                )

            if method in ("spearman", "both"):
                r_s, p_s = stats.spearmanr(col, y_np)
                row["spearman_r"] = float(r_s)
                row["spearman_p"] = float(p_s)
                row["spearman_significant"] = bool(
                    p_s < significance_threshold
                )

            row["eta_squared"] = nan
            row["anova_p"] = nan
            row["anova_significant"] = nan

        records.append(row)

    result = pd.DataFrame(records, index=pd.Index(X.columns, name="feature"))

    # Ranking: use sqrt(eta_squared) for categorical (correlation ratio η,
    # range [0,1]) and max(|r|) for continuous — both on [0,1] scale.
    abs_scores = pd.Series(index=X.columns, dtype=float)
    for feature in X.columns:
        if feature in cat_set:
            eta2 = result.loc[feature, "eta_squared"]
            abs_scores[feature] = (
                math.sqrt(float(eta2)) if not math.isnan(float(eta2)) else 0.0
            )
        else:
            if method == "both":
                abs_scores[feature] = (
                    result.loc[feature, ["pearson_r", "spearman_r"]]
                    .abs()
                    .max()
                )
            elif method == "pearson":
                abs_scores[feature] = abs(float(result.loc[feature, "pearson_r"]))
            else:
                abs_scores[feature] = abs(float(result.loc[feature, "spearman_r"]))

    result["rank"] = (
        abs_scores.rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )

    n_cat = len(cat_set & set(X.columns))
    logger.info(
        "Correlations (%s): %d continuous + %d categorical features analyzed.",
        method,
        len(X.columns) - n_cat,
        n_cat,
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _eta_squared(groups: list[np.ndarray], y_all: np.ndarray) -> float:
    """Compute eta-squared (η²) effect size for one-way ANOVA.

    η² = SS_between / SS_total

    Parameters
    ----------
    groups : list of np.ndarray
        Per-group arrays of the target variable.
    y_all : np.ndarray
        Full target array (used for SS_total).

    Returns
    -------
    float
        eta-squared in [0, 1].
    """
    grand_mean = y_all.mean()
    ss_total = float(np.sum((y_all - grand_mean) ** 2))
    if ss_total == 0.0:
        return 0.0
    ss_between = float(
        sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    )
    return max(0.0, min(1.0, ss_between / ss_total))


def _validate_method(method: str) -> None:
    """Raise ValueError if *method* is not a valid CorrelationMethod."""
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid options: {sorted(_VALID_METHODS)}"
        )
