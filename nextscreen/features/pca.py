"""PCA-based feature analysis: explained variance and component loadings."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def run_pca(
    X: pd.DataFrame,
    variance_threshold: float = 0.90,
    max_components: int = 5,
) -> dict[str, object]:
    """Run PCA and return explained variance and loading information.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features). Will be
        standardized (zero mean, unit variance) before fitting.
    variance_threshold : float, optional
        Cumulative explained-variance ratio used to select the number of
        components to retain. Default is 0.90.
    max_components : int, optional
        Hard upper bound on the number of components returned regardless
        of *variance_threshold*. Default is 5.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'explained_variance_ratio'`` : np.ndarray
            Per-component explained-variance ratios for retained PCs.
        ``'cumulative_variance'`` : np.ndarray
            Cumulative explained-variance ratios for retained PCs.
        ``'n_components'`` : int
            Number of components selected.
        ``'loadings'`` : pd.DataFrame
            Loading matrix of shape (n_features, n_components) with
            feature names as the index and ``'PC1'``, ``'PC2'``, ...
            as column names.
        ``'feature_rank'`` : pd.DataFrame
            Features ranked by their maximum absolute loading across
            the retained components, with columns
            ``['feature', 'max_loading', 'rank']``.

    Raises
    ------
    ValueError
        If *X* is empty, if *variance_threshold* is not in (0, 1], or
        if *max_components* is less than 1.
    """
    if X.empty:
        raise ValueError("X must contain at least one row.")
    if not (0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1]; "
            f"got {variance_threshold}."
        )
    if max_components < 1:
        raise ValueError(
            f"max_components must be at least 1; got {max_components}."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # sklearn requires n_components <= min(n_samples - 1, n_features).
    max_possible = min(len(X) - 1, X.shape[1])
    n_fit = min(max_components, max_possible)

    pca = PCA(n_components=n_fit)
    pca.fit(X_scaled)

    evr: np.ndarray = pca.explained_variance_ratio_
    cumulative: np.ndarray = np.cumsum(evr)

    # Smallest number of components reaching variance_threshold.
    idx = int(np.searchsorted(cumulative, variance_threshold))
    n_selected = min(idx + 1, n_fit)

    pc_labels = [f"PC{i + 1}" for i in range(n_selected)]

    # pca.components_ has shape (n_fit, n_features); transpose to
    # (n_features, n_selected) for the loadings DataFrame.
    loadings = pd.DataFrame(
        pca.components_[:n_selected].T,
        index=X.columns,
        columns=pc_labels,
    )

    # Project all observations into the reduced PC space.
    scores_arr = pca.transform(X_scaled)[:, :n_selected]
    scores_df = pd.DataFrame(scores_arr, index=X.index, columns=pc_labels)

    max_abs_loading = loadings.abs().max(axis=1)
    feature_rank = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "max_loading": max_abs_loading.values,
        }
    )
    feature_rank = feature_rank.sort_values(
        "max_loading", ascending=False
    ).reset_index(drop=True)
    feature_rank["rank"] = (
        feature_rank["max_loading"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    logger.info(
        "PCA: %d component(s) selected (cumulative variance=%.3f).",
        n_selected,
        float(cumulative[n_selected - 1]),
    )

    return {
        "explained_variance_ratio": evr[:n_selected],
        "cumulative_variance": cumulative[:n_selected],
        "n_components": n_selected,
        "loadings": loadings,
        "scores": scores_df,
        "feature_rank": feature_rank,
    }
