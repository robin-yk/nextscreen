"""Format feature bounds for NEXTorch and run Bayesian Optimization."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import torch
from nextorch import bo
from nextorch.parameter import Parameter

logger = logging.getLogger(__name__)

VariableType = Literal["continuous", "categorical", "integer"]

# Ordinal integer parameters: interval between levels.
_INTEGER_INTERVAL = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_scalarized_target(
    df: pd.DataFrame,
    target_cols: list[str],
    weights: list[float],
    col_name: str = "_combined_score",
) -> pd.DataFrame:
    """Return a copy of *df* with a new scalarized target column.

    Each target is independently normalized to [0, 1] using its
    observed min–max range, then a weighted sum is computed.  This
    allows single-objective BO to approximate multi-objective
    optimization by running at several preset weight combinations
    (e.g. maximize t1, maximize t2, balanced).

    Parameters
    ----------
    df : pd.DataFrame
        Training data containing all columns named in *target_cols*.
    target_cols : list of str
        Names of the target columns to combine.
    weights : list of float
        Non-negative weights, one per target.  They are automatically
        normalized to sum to 1, so ``[1, 0]`` and ``[10, 0]`` are
        equivalent.
    col_name : str, optional
        Name of the new combined-score column.  Default
        ``"_combined_score"``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional column *col_name* containing
        the weighted normalized score.

    Raises
    ------
    ValueError
        If *target_cols* or *weights* is empty, if their lengths differ,
        or if any weight is negative.
    """
    if not target_cols:
        raise ValueError("target_cols must not be empty.")
    if len(weights) != len(target_cols):
        raise ValueError(
            f"len(weights)={len(weights)} must equal "
            f"len(target_cols)={len(target_cols)}."
        )

    w_arr = np.array(weights, dtype=float)
    if np.any(w_arr < 0):
        raise ValueError("All weights must be non-negative.")
    w_sum = w_arr.sum()
    if w_sum == 0:
        raise ValueError("At least one weight must be positive.")
    w_arr = w_arr / w_sum

    df_out = df.copy()
    score = np.zeros(len(df), dtype=float)
    for col, w in zip(target_cols, w_arr):
        y = df[col].to_numpy(dtype=float)
        y_min, y_max = y.min(), y.max()
        if y_max > y_min:
            y_norm = (y - y_min) / (y_max - y_min)
        else:
            y_norm = np.full_like(y, 0.5)
        score += w * y_norm

    df_out[col_name] = score
    logger.info(
        "Scalarized %d targets → '%s' "
        "(weights: %s).",
        len(target_cols),
        col_name,
        [f"{w:.3f}" for w in w_arr],
    )
    return df_out


def build_parameter_space(
    selected_features: list[str],
    bounds: dict[str, dict[str, object]],
) -> list[Parameter]:
    """Translate user-defined feature bounds into a list of Parameters.

    Parameters
    ----------
    selected_features : list of str
        Feature names chosen by the user to optimize over.
    bounds : dict
        Nested dict mapping each feature name to its bound specification::

            {
                "temperature": {
                    "lower": 100.0,
                    "upper": 300.0,
                    "type": "continuous",
                },
                "catalyst": {
                    "lower": 0,
                    "upper": 3,
                    "type": "integer",
                },
            }

    Returns
    -------
    list of Parameter
        One :class:`~nextorch.parameter.Parameter` per feature, ready to
        pass directly to :func:`bo.Experiment.define_space`.

    Raises
    ------
    ValueError
        If *selected_features* is empty, or if any feature's lower bound
        is >= its upper bound.
    """
    if not selected_features:
        raise ValueError(
            "selected_features must contain at least one feature."
        )

    params: list[Parameter] = []
    for feat in selected_features:
        b = bounds.get(feat, {})
        lower = float(b.get("lower", 0.0))
        upper = float(b.get("upper", 1.0))
        vtype = str(b.get("type", "continuous"))

        if vtype == "integer":
            # NEXTorch ordinal requires interval + x_range (not values alone).
            if lower >= upper:
                raise ValueError(
                    f"Feature '{feat}': lower bound ({lower}) "
                    f"must be less than upper bound ({upper})."
                )
            step = float(b.get("step") or _INTEGER_INTERVAL)
            param = Parameter(
                name=feat,
                x_type="ordinal",
                x_range=[lower, upper],
                interval=step,
            )
        elif vtype == "categorical":
            cat_values = b.get("values")
            if cat_values and len(cat_values) >= 2:
                param = Parameter(
                    name=feat,
                    x_type="categorical",
                    values=np.array(cat_values),
                )
            else:
                # No discrete values supplied — fall back to continuous.
                if lower >= upper:
                    raise ValueError(
                        f"Feature '{feat}': lower bound ({lower}) "
                        f"must be less than upper bound ({upper})."
                    )
                logger.warning(
                    "Feature '%s' is marked 'categorical' but no "
                    "discrete values were provided; treating as "
                    "continuous over [%.4g, %.4g].",
                    feat,
                    lower,
                    upper,
                )
                param = Parameter(
                    name=feat,
                    x_type="continuous",
                    x_range=[lower, upper],
                )
        else:
            if lower >= upper:
                raise ValueError(
                    f"Feature '{feat}': lower bound ({lower}) "
                    f"must be less than upper bound ({upper})."
                )
            param = Parameter(
                name=feat,
                x_type="continuous",
                x_range=[lower, upper],
            )

        params.append(param)

    return params


def run_optimization(
    parameters: list[Parameter],
    training_data: pd.DataFrame,
    target_col: str,
    n_suggestions: int = 5,
    random_state: int = 42,
    return_ard_importance: bool = False,
    aux_target_cols: list[str] | None = None,
    categorical_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run one round of Bayesian Optimization and return suggested experiments.

    Parameters
    ----------
    parameters : list of Parameter
        NEXTorch parameter list built by :func:`build_parameter_space`.
    training_data : pd.DataFrame
        Historical experiment data (features + target) used to condition the
        surrogate model.
    target_col : str
        Name of the target column in *training_data* to optimize.
    n_suggestions : int, optional
        Number of new experimental conditions to suggest. Must be in [1, 20].
        Default is 5.
    random_state : int, optional
        Random seed passed to PyTorch and NumPy for reproducibility.
        Default is 42.
    return_ard_importance : bool, optional
        If ``True``, also return the ARD lengthscales from the fitted
        GP surrogate as a second DataFrame.  Default ``False``.
    aux_target_cols : list of str or None, optional
        Additional target columns to predict at the suggested points.
        Useful for multi-objective BO where *target_col* is a scalarized
        combined score — pass the original target names here to get
        interpretable per-target predictions (in original units) alongside
        the suggested experimental conditions.  When provided, the
        ``'predicted_<target_col>'`` (combined-score) column is omitted and
        replaced with ``'predicted_<aux>'`` columns for each aux target.
        Default ``None``.
    categorical_maps : dict or None, optional
        Mapping ``{feature_name: {label: integer_code}}`` as returned by
        :func:`~nextscreen.data.loader.encode_categoricals`.  When provided,
        integer-encoded categorical columns in the output are replaced with
        their original string labels (e.g. ``0 → "Ni"``).
        Default ``None`` (codes are left as-is).

    Returns
    -------
    pd.DataFrame
        Table of suggested experiments with columns for each feature, an
        ``'uncertainty'`` column (approximate 1-sigma standard deviation
        from the 95 % CI of the optimized GP), and either:

        - a ``'predicted_<target_col>'`` column when *aux_target_cols* is
          ``None``, or
        - one ``'predicted_<aux>'`` column per entry in *aux_target_cols*
          (GP posterior means in original target units).

        Shape: (n_suggestions, n_features + 1 + n_predictions).
    ard_df : pd.DataFrame or None
        Only returned when *return_ard_importance* is ``True``.
        Columns: ``['feature', 'gp_lengthscale', 'gp_importance']``,
        sorted by descending ``gp_importance``.  ``None`` if extraction
        fails.

    Raises
    ------
    ValueError
        If *n_suggestions* is not in [1, 20], if any feature column is
        missing from *training_data*, if *target_col* is not found, or if
        *training_data* has fewer than two rows.
    """
    if not (1 <= n_suggestions <= 20):
        raise ValueError(
            f"n_suggestions must be in [1, 20]; got {n_suggestions}."
        )

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    feature_names: list[str] = [p.name for p in parameters]

    # Validate columns.
    missing = [
        f for f in feature_names
        if f not in training_data.columns
    ]
    if missing:
        raise ValueError(
            f"Features missing from training_data: {missing}"
        )
    if target_col not in training_data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found "
            "in training_data."
        )

    X_real = training_data[feature_names].to_numpy(dtype=float)
    Y_real = training_data[[target_col]].to_numpy(dtype=float)

    if X_real.shape[0] < 2:
        raise ValueError(
            "training_data must contain at least 2 rows "
            "to fit a Gaussian Process surrogate."
        )

    # Ensure torch can see numpy before running BO.
    # On some platforms torch._numpy_available is set to False if torch is
    # first imported before numpy (e.g. via a Streamlit internal import).
    # By this point numpy is definitely loaded, so correct the flag.
    if hasattr(torch, '_numpy_available') and not torch._numpy_available:
        try:
            np.array([1.0])
            torch._numpy_available = True
        except Exception:
            pass

    # Set up the NEXTorch Experiment.
    exp = bo.Experiment(name="nextscreen_bo")
    exp.define_space(parameters)
    exp.input_data(
        X_real,
        Y_real,
        X_names=feature_names,
        Y_names=[target_col],
    )
    # Maximise by default; users can explicitly target minimisation
    # by negating their response before calling this function.
    exp.set_optim_specs(maximize=True)

    # Analytic EI for single point; Monte-Carlo qEI for batches.
    acq_name = "qEI" if n_suggestions > 1 else "EI"
    X_new, X_new_real, _ = exp.generate_next_point(
        acq_func_name=acq_name,
        n_candidates=n_suggestions,
    )

    # Predictions in real scale with 95 % confidence intervals.
    Y_pred, Y_lower, Y_upper = exp.predict_real(
        X_new_real, show_confidence=True
    )

    # Approximate 1-sigma from 95 % CI: std ≈ (upper − lower) / (2 × 1.96)
    uncertainty = (Y_upper - Y_lower) / (2.0 * 1.96)

    result = pd.DataFrame(X_new_real, columns=feature_names)

    # When aux targets are provided, omit the (often uninterpretable)
    # combined-score prediction and replace it with per-target predictions
    # in original units.
    if not aux_target_cols:
        result[f"predicted_{target_col}"] = Y_pred.flatten()
    result["uncertainty"] = uncertainty.flatten()

    # Fit one lightweight GP per aux target and predict at the suggested X.
    # make_scalarized_target returns a copy that still contains the original
    # target columns, so they are available in training_data here.
    if aux_target_cols:
        for aux_col in aux_target_cols:
            if aux_col not in training_data.columns:
                logger.warning(
                    "aux_target_col '%s' not found in training_data; "
                    "skipping individual prediction.",
                    aux_col,
                )
                continue
            try:
                aux_exp = bo.Experiment(name=f"nextscreen_aux_{aux_col}")
                aux_exp.define_space(parameters)
                aux_exp.input_data(
                    X_real,
                    training_data[[aux_col]].to_numpy(dtype=float),
                    X_names=feature_names,
                    Y_names=[aux_col],
                )
                aux_exp.set_optim_specs(maximize=True)
                # generate_next_point fits the GP surrogate; the returned
                # candidate is discarded — we only need the trained model.
                aux_exp.generate_next_point(
                    acq_func_name="EI",
                    n_candidates=1,
                )
                Y_aux, _, _ = aux_exp.predict_real(
                    X_new_real, show_confidence=True
                )
                result[f"predicted_{aux_col}"] = Y_aux.flatten()
                logger.info(
                    "Aux GP: predicted '%s' at %d suggested points.",
                    aux_col,
                    n_suggestions,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not predict aux target '%s': %s", aux_col, exc
                )

    logger.info(
        "NEXTorch BO: %d suggestion(s) generated for '%s' "
        "(acq=%s).",
        n_suggestions,
        target_col,
        acq_name,
    )

    if categorical_maps:
        result = _decode_categoricals(result, categorical_maps)

    if not return_ard_importance:
        return result

    # Extract ARD lengthscales from the fitted GP surrogate.
    # NEXTorch normalises X to [0, 1] before fitting, so lengthscales
    # are directly comparable across features.
    ard_df: pd.DataFrame | None = None
    try:
        ls = (
            exp.model.covar_module.base_kernel.lengthscale
            .detach()
            .squeeze()
            .numpy()
        )
        ls = np.atleast_1d(ls).astype(float)
        importance_raw = 1.0 / ls
        importance = importance_raw / importance_raw.sum()
        ard_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "gp_lengthscale": ls.tolist(),
                    "gp_importance": importance.tolist(),
                }
            )
            .sort_values("gp_importance", ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            "BO ARD: top feature = '%s' "
            "(lengthscale=%.4f).",
            ard_df.iloc[0]["feature"],
            ard_df.iloc[0]["gp_lengthscale"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not extract ARD lengthscales from BO GP: %s", exc
        )

    return result, ard_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _decode_categoricals(
    result: pd.DataFrame,
    categorical_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Replace integer-encoded categorical columns with their original labels.

    Parameters
    ----------
    result : pd.DataFrame
        Suggestions DataFrame whose categorical feature columns contain
        integer codes (as produced by
        :func:`~nextscreen.data.loader.encode_categoricals`).
    categorical_maps : dict
        Mapping ``{feature_name: {label: integer_code}}`` as returned by
        :func:`~nextscreen.data.loader.encode_categoricals`.

    Returns
    -------
    pd.DataFrame
        Copy of *result* with categorical columns replaced by string labels.
    """
    out = result.copy()
    for col, lmap in categorical_maps.items():
        if col not in out.columns:
            continue
        decode = {int(v): k for k, v in lmap.items()}
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .round()
            .astype("Int64")
            .map(decode)
        )
    return out



def run_pareto_optimization(
    parameters: list[Parameter],
    training_data: pd.DataFrame,
    target_cols: list[str],
    n_suggestions: int = 5,
    random_state: int = 42,
    ref_point: list[float] | None = None,
    categorical_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Run multi-objective Bayesian Optimization using qEHVI.

    Uses q-Expected Hypervolume Improvement (qEHVI) [Daulton2020]_ to
    jointly optimize all target columns simultaneously.  Returns a batch
    of suggestions that maximally expands the hypervolume dominated by the
    current Pareto front — the suggestions collectively represent different
    trade-offs between the objectives rather than committing to a single
    weight combination.

    Unlike :func:`run_optimization` with :func:`make_scalarized_target`,
    this function does **not** require the user to specify weights.  The
    Pareto-front approach is preferable when the relative importance of
    objectives is unknown or when you want to discover the full trade-off
    surface.

    Parameters
    ----------
    parameters : list of Parameter
        NEXTorch parameter list built by :func:`build_parameter_space`.
    training_data : pd.DataFrame
        Historical experiment data containing all feature and target columns.
    target_cols : list of str
        Names of the target columns to optimize jointly.  Must have ≥ 2.
    n_suggestions : int, optional
        Number of candidate experiments to suggest.  Must be in [1, 20].
        Default is 5.
    random_state : int, optional
        Random seed for reproducibility.  Default 42.
    ref_point : list of float or None, optional
        Reference point in objective space (one value per target).  Points
        that do not dominate the reference point are not considered when
        computing hypervolume improvement.  If ``None``, automatically set
        to 10 % below the worst observed value on each objective.
        Default ``None``.
    categorical_maps : dict or None, optional
        Mapping ``{feature_name: {label: integer_code}}`` as returned by
        :func:`~nextscreen.data.loader.encode_categoricals`.  When provided,
        integer-encoded categorical columns in the output are replaced with
        their original string labels (e.g. ``0 → "Ni"``).
        Default ``None`` (codes are left as-is).

    Returns
    -------
    pd.DataFrame
        Suggested experiments with columns for each feature and one
        ``'predicted_<target>'`` column per entry in *target_cols*
        (GP posterior mean in original target units).
        Shape: (n_suggestions, n_features + n_targets).

    Raises
    ------
    ValueError
        If fewer than 2 targets are supplied, *n_suggestions* is out of
        [1, 20], any feature/target column is missing from *training_data*,
        or *training_data* has fewer than 2 rows.
    """
    if len(target_cols) < 2:
        raise ValueError(
            "target_cols must contain at least 2 targets for Pareto BO; "
            f"got {len(target_cols)}."
        )
    if not (1 <= n_suggestions <= 20):
        raise ValueError(
            f"n_suggestions must be in [1, 20]; got {n_suggestions}."
        )

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    feature_names: list[str] = [p.name for p in parameters]
    X_ranges = [[p.x_range[0], p.x_range[1]] for p in parameters]

    missing = [f for f in feature_names if f not in training_data.columns]
    if missing:
        raise ValueError(f"Features missing from training_data: {missing}")
    missing_targets = [
        t for t in target_cols if t not in training_data.columns
    ]
    if missing_targets:
        raise ValueError(
            f"Targets missing from training_data: {missing_targets}"
        )

    X_real = training_data[feature_names].to_numpy(dtype=float)
    Y_real = training_data[target_cols].to_numpy(dtype=float)

    if X_real.shape[0] < 2:
        raise ValueError(
            "training_data must contain at least 2 rows to fit a GP surrogate."
        )

    # Reference point: 10 % below the worst observed value per objective.
    if ref_point is None:
        y_min = Y_real.min(axis=0)
        y_range = Y_real.max(axis=0) - y_min
        y_range = np.where(y_range > 0, y_range, np.ones_like(y_range))
        ref_point = (y_min - 0.1 * y_range).tolist()

    if hasattr(torch, '_numpy_available') and not torch._numpy_available:
        try:
            np.array([1.0])
            torch._numpy_available = True
        except Exception:
            pass

    exp = bo.EHVIMOOExperiment("nextscreen_pareto")
    exp.input_data(
        X_real,
        Y_real,
        X_ranges=X_ranges,
        X_names=feature_names,
        Y_names=target_cols,
        unit_flag=True,
    )
    exp.set_ref_point(ref_point)
    exp.set_optim_specs(maximize=True)

    X_new, X_new_real, _ = exp.generate_next_point(
        n_candidates=n_suggestions
    )

    result = pd.DataFrame(X_new_real, columns=feature_names)

    # Predict each objective at the suggested points.
    try:
        Y_pred, Y_lower, Y_upper = exp.predict_real(
            X_new_real, show_confidence=True
        )
        Y_pred = np.atleast_2d(Y_pred)
        Y_lower = np.atleast_2d(Y_lower)
        Y_upper = np.atleast_2d(Y_upper)
        for i, tcol in enumerate(target_cols):
            result[f"predicted_{tcol}"] = Y_pred[:, i]
            result[f"uncertainty_{tcol}"] = (
                Y_upper[:, i] - Y_lower[:, i]
            ) / (2.0 * 1.96)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not get predictions for Pareto suggestions: %s", exc
        )

    logger.info(
        "Pareto BO (EHVIMOOExperiment): %d suggestion(s) across %d "
        "objectives (%s).",
        n_suggestions,
        len(target_cols),
        ", ".join(target_cols),
    )

    if categorical_maps:
        result = _decode_categoricals(result, categorical_maps)

    return result
