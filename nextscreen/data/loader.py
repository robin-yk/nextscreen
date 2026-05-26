"""CSV and Excel data loading with replicate detection and handling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

ReplicateStrategy = Literal["average", "keep_all", "std_uncertainty"]

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_file(file_path: Path | str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    file_path : Path or str
        Path to the CSV (.csv) or Excel (.xlsx / .xls) file.

    Returns
    -------
    pd.DataFrame
        Raw data as loaded from disk, with no preprocessing applied.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file extension is not supported, or if the file is empty
        (contains no rows and no columns).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Loading %s file: %s", suffix.lstrip(".").upper(), path)

    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if df.empty and df.columns.empty:
        raise ValueError(f"File is empty (no columns found): {path}")

    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
    return df


def detect_replicates(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Identify rows with identical input-feature values (replicates).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature and target columns.
    feature_cols : list of str
        Column names that define an experimental condition.
        Rows sharing the same values in these columns are replicates.

    Returns
    -------
    pd.DataFrame
        Subset of *df* containing only the replicate rows, with an added
        ``replicate_group`` integer column (0-based) identifying each group.
        Returns an empty DataFrame (same columns as *df* plus
        ``replicate_group``) if no replicates are found.

    Raises
    ------
    ValueError
        If *feature_cols* is empty, or if any name in *feature_cols* is
        absent from *df*.
    """
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column name.")

    _validate_columns(df, feature_cols, label="feature_cols")

    # Mark every row that belongs to a group with more than one member.
    mask = df.duplicated(subset=feature_cols, keep=False)
    replicate_rows = df[mask].copy()

    if replicate_rows.empty:
        # Return schema-correct empty frame so callers can check `.empty`.
        empty = replicate_rows.copy()
        empty["replicate_group"] = pd.Series(dtype="int64")
        return empty

    # Assign stable group IDs (ngroup is order-of-first-appearance in df).
    all_group_ids = df.groupby(feature_cols, sort=False).ngroup()
    raw_ids = all_group_ids[mask]

    # Re-index to 0-based contiguous integers among replicate groups only.
    unique_sorted = sorted(raw_ids.unique())
    remap = {old: new for new, old in enumerate(unique_sorted)}
    replicate_rows["replicate_group"] = raw_ids.map(remap).astype("int64")

    logger.info(
        "Detected %d replicate rows across %d group(s).",
        len(replicate_rows),
        replicate_rows["replicate_group"].nunique(),
    )
    return replicate_rows


def handle_replicates(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    strategy: ReplicateStrategy = "average",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate or retain replicate rows according to the chosen strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, may contain replicates.
    feature_cols : list of str
        Input feature column names (used to group replicates).
    target_cols : list of str
        Target variable column names to aggregate.
    strategy : {'average', 'keep_all', 'std_uncertainty'}, optional
        How to handle replicate rows:

        - ``'average'`` : replace replicates with their mean (default).
        - ``'keep_all'`` : return all rows unchanged.
        - ``'std_uncertainty'`` : return the mean and add a ``<col>_std``
          column containing the standard deviation for each target.

    Returns
    -------
    processed : pd.DataFrame
        Dataset after applying the chosen replicate strategy. Row index is
        reset to 0-based integers. Column order: feature_cols, target_cols,
        [<target>_std …] (only for ``'std_uncertainty'``), remaining columns.
    replicate_summary : pd.DataFrame
        Group-level summary of detected replicate conditions. Empty DataFrame
        if no replicates were found. Columns: feature_cols,
        ``replicate_group``, ``n_replicates``, and per-target
        ``<col>_mean`` / ``<col>_std`` columns.

    Raises
    ------
    ValueError
        If *feature_cols* or *target_cols* is empty, if any column name is
        absent from *df*, if feature_cols and target_cols overlap, or if
        *strategy* is not one of the accepted literals.
    """
    _validate_strategy(strategy)
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one column name.")
    if not target_cols:
        raise ValueError("target_cols must contain at least one column name.")
    _validate_columns(df, feature_cols, label="feature_cols")
    _validate_columns(df, target_cols, label="target_cols")

    overlap = set(feature_cols) & set(target_cols)
    if overlap:
        raise ValueError(
            "The following columns appear in both feature_cols and "
            f"target_cols: {sorted(overlap)}"
        )

    replicate_df = detect_replicates(df, feature_cols)
    replicate_summary = _build_replicate_summary(
        replicate_df, feature_cols, target_cols
    )

    if strategy == "keep_all":
        logger.info(
            "Replicate strategy: keep_all — %d rows returned unchanged.",
            len(df),
        )
        return df.copy().reset_index(drop=True), replicate_summary

    # --- aggregate strategies (average / std_uncertainty) ---
    other_cols = [
        c for c in df.columns
        if c not in feature_cols and c not in target_cols
    ]

    # Pass 1: means for targets + first-value for any remaining columns.
    agg_mean: dict[str, str] = {col: "mean" for col in target_cols}
    agg_mean.update({col: "first" for col in other_cols})

    processed = (
        df.groupby(feature_cols, sort=False)
        .agg(agg_mean)
        .reset_index()
    )

    if strategy == "std_uncertainty":
        # Pass 2: per-target std dev (NaN for single-member groups).
        std_df = (
            df.groupby(feature_cols, sort=False)
            .agg({col: "std" for col in target_cols})
            .reset_index()
            .rename(columns={col: f"{col}_std" for col in target_cols})
        )
        std_cols = [f"{col}_std" for col in target_cols]
        processed = processed.merge(
            std_df[feature_cols + std_cols], on=feature_cols
        )

    # Enforce consistent column order.
    col_order = list(feature_cols) + list(target_cols)
    if strategy == "std_uncertainty":
        col_order += [f"{col}_std" for col in target_cols]
    col_order += other_cols
    processed = processed[col_order].reset_index(drop=True)

    logger.info(
        "Replicate strategy: %s — %d rows → %d rows.",
        strategy,
        len(df),
        len(processed),
    )
    return processed, replicate_summary


def drop_missing_rows(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Drop rows that contain missing values and report which were removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset, may contain NaN entries.
    cols : list of str or None, optional
        Columns to check for missing values.  If *None*, all columns are
        checked.  Pass ``feature_cols + target_cols`` to ignore unrelated
        columns.

    Returns
    -------
    cleaned : pd.DataFrame
        Copy of *df* with missing-value rows removed.  Original index is
        reset to 0-based integers.
    dropped_indices : list of int
        Original (0-based) row indices that were removed.  Empty list if no
        rows were dropped.
    """
    check_cols = cols if cols is not None else list(df.columns)
    mask = df[check_cols].isna().any(axis=1)
    dropped_indices = list(df.index[mask])

    if dropped_indices:
        logger.warning(
            "Dropped %d row(s) with missing values (original indices: %s).",
            len(dropped_indices),
            dropped_indices,
        )

    cleaned = df[~mask].reset_index(drop=True)
    return cleaned, dropped_indices


def encode_categoricals(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Detect and encode string/object columns to integer codes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that may contain string or object-dtype columns.
    cols : list of str, optional
        Columns to encode.  If *None*, every column whose dtype is
        ``object`` is auto-detected and encoded.

    Returns
    -------
    encoded : pd.DataFrame
        Copy of *df* with each target column replaced by integer codes
        (0, 1, 2, … ordered alphabetically by original label).
    maps : dict[str, dict[str, int]]
        ``{column_name: {original_label: integer_code}}`` for every
        encoded column.  Empty dict when no object columns are found.
    """
    if cols is None:
        cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]

    if not cols:
        return df.copy(), {}

    encoded = df.copy()
    maps: dict[str, dict[str, int]] = {}

    for col in cols:
        codes, uniques = pd.factorize(df[col], sort=True)
        encoded[col] = codes
        maps[col] = {
            str(label): int(idx)
            for idx, label in enumerate(uniques)
        }
        logger.info(
            "Encoded column '%s': %d categories → codes 0…%d.",
            col,
            len(uniques),
            len(uniques) - 1,
        )

    return encoded, maps


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    """Raise ValueError if any column in *cols* is absent from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns listed in {label} not found in DataFrame: {missing}"
        )


def _validate_strategy(strategy: str) -> None:
    """Raise ValueError if *strategy* is not a recognized ReplicateStrategy."""
    valid = {"average", "keep_all", "std_uncertainty"}
    if strategy not in valid:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid options: {sorted(valid)}"
        )


def _build_replicate_summary(
    replicate_df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame:
    """Build a group-level summary from :func:`detect_replicates` output.

    Parameters
    ----------
    replicate_df : pd.DataFrame
        Output of :func:`detect_replicates` (may be empty).
    feature_cols : list of str
        Feature column names that define the experimental condition.
    target_cols : list of str
        Target column names for which to compute per-group mean and std.

    Returns
    -------
    pd.DataFrame
        One row per replicate group. Columns: feature_cols,
        ``replicate_group``, ``n_replicates``, ``<target>_mean``,
        ``<target>_std`` for each target present in *replicate_df*.
        Empty DataFrame (with the same schema) if *replicate_df* is empty.
    """
    base_cols = feature_cols + ["replicate_group", "n_replicates"]
    stat_cols = [
        f"{t}_{stat}" for t in target_cols for stat in ("mean", "std")
    ]
    schema_cols = base_cols + [c for c in stat_cols if c not in base_cols]

    if replicate_df.empty:
        return pd.DataFrame(columns=schema_cols)

    rows: list[dict[str, object]] = []
    for group_id, group in replicate_df.groupby("replicate_group", sort=True):
        row: dict[str, object] = {
            col: group[col].iloc[0] for col in feature_cols
        }
        row["replicate_group"] = int(group_id)
        row["n_replicates"] = len(group)
        for tcol in target_cols:
            if tcol in group.columns:
                row[f"{tcol}_mean"] = group[tcol].mean()
                row[f"{tcol}_std"] = group[tcol].std()
        rows.append(row)

    return pd.DataFrame(rows, columns=schema_cols)
