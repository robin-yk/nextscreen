"""ARD-GP feature importance via BoTorch SingleTaskGP lengthscales."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_ard_gp(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit an ARD Gaussian Process and rank features by lengthscale.

    A ``SingleTaskGP`` with per-feature (ARD) Matérn-5/2 lengthscales is
    fitted on the data.  Features with *shorter* lengthscales cause the
    GP response surface to vary more rapidly in that direction and are
    therefore considered more important.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : pd.Series
        Target variable with length n_samples.
    n_iter : int, optional
        Maximum L-BFGS-B iterations for GP hyper-parameter optimization.
        Default 100.
    random_state : int, optional
        Seed for PyTorch.  Default 42.

    Returns
    -------
    pd.DataFrame
        One row per feature, sorted by descending ``importance``.
        Columns: ``['feature', 'lengthscale', 'importance', 'rank']``.
        ``importance`` is the inverse lengthscale normalized to sum to 1.

    Raises
    ------
    ValueError
        If *X* is empty, *X* and *y* have different lengths, or there
        are fewer rows than features + 2 (GP would be under-determined).
    RuntimeError
        If GP hyper-parameter optimization fails.

    Notes
    -----
    X is min-max scaled to [0, 1] and y is standardized before fitting,
    matching the internal normalization used by NEXTorch / BoTorch.
    Lengthscales are therefore directly comparable across features.

    Categorical and integer columns encoded as 0/1/2/… integers are
    supported numerically, but their lengthscales are less interpretable
    because the ordinal spacing is arbitrary.
    """
    import torch  # deferred: keep Streamlit startup fast
    from botorch.models import SingleTaskGP
    from botorch.optim.fit import fit_gpytorch_torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    if X.empty:
        raise ValueError("X must contain at least one row.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length ({len(X)} vs {len(y)})."
        )
    n_samples, n_features = X.shape
    if n_samples < n_features + 2:
        raise ValueError(
            f"Need at least n_features + 2 = {n_features + 2} rows to fit "
            f"an ARD-GP; got {n_samples}."
        )

    torch.manual_seed(random_state)

    # ── Normalise ────────────────────────────────────────────────────────────
    X_arr = X.to_numpy(dtype=float)
    col_min = X_arr.min(axis=0)
    col_max = X_arr.max(axis=0)
    col_range = col_max - col_min
    # Constant columns → set to 0.5, warn.
    zero_range = col_range == 0.0
    if zero_range.any():
        zero_cols = [X.columns[i] for i in np.where(zero_range)[0]]
        logger.warning(
            "ARD-GP: constant columns will be set to 0.5: %s", zero_cols
        )
        col_range[zero_range] = 1.0  # avoid divide-by-zero
    X_scaled = (X_arr - col_min) / col_range
    X_scaled[:, zero_range] = 0.5

    y_arr = y.to_numpy(dtype=float)
    y_std = y_arr.std()
    if y_std == 0.0:
        raise ValueError("Target y has zero variance; cannot fit GP.")
    y_scaled = (y_arr - y_arr.mean()) / y_std

    X_t = torch.tensor(X_scaled, dtype=torch.double)
    Y_t = torch.tensor(y_scaled, dtype=torch.double).unsqueeze(1)

    # ── Fit GP ───────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = SingleTaskGP(train_X=X_t, train_Y=Y_t)
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        try:
            fit_gpytorch_torch(mll, options={"maxiter": n_iter})
        except Exception as exc:
            raise RuntimeError(
                f"ARD-GP hyper-parameter optimization failed: {exc}"
            ) from exc
        model.eval()

    # ── Extract lengthscales ─────────────────────────────────────────────────
    ls = (
        model.covar_module.base_kernel.lengthscale
        .detach()
        .squeeze()
        .numpy()
    )
    ls = np.atleast_1d(ls).astype(float)

    importance_raw = 1.0 / ls
    importance = importance_raw / importance_raw.sum()

    result = pd.DataFrame(
        {
            "feature": list(X.columns),
            "lengthscale": ls.tolist(),
            "importance": importance.tolist(),
        }
    )
    result = (
        result.sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    result["rank"] = (
        result["importance"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    logger.info(
        "ARD-GP: top feature = '%s' "
        "(lengthscale=%.4f, importance=%.4f).",
        result.iloc[0]["feature"],
        result.iloc[0]["lengthscale"],
        result.iloc[0]["importance"],
    )
    return result
