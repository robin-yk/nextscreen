"""Rule-based plain-English interpretation of feature selection results."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_CAVEAT = (
    "These are data-driven suggestions. "
    "Domain expertise should guide final decisions."
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _join_names(names: list[str]) -> str:
    """Format a list of feature names as readable text."""
    if not names:
        return ""
    if len(names) == 1:
        return f"'{names[0]}'"
    quoted = [f"'{n}'" for n in names]
    return ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def _pearson_narrative(
    result: pd.DataFrame,
    target_name: str,
    parts: list[str],
) -> None:
    """Append Pearson correlation sentences to *parts*.

    Rows where ``pearson_r`` is NaN (categorical features assessed via
    ANOVA) are silently skipped.
    """
    # Only consider continuous features (categorical rows have NaN r-values).
    cont = result[result["pearson_r"].notna()]
    if cont.empty:
        return
    sig = cont[cont["pearson_significant"].astype(bool)]
    no_linear = cont[
        (cont["pearson_r"].abs() < 0.1)
        & (cont["pearson_p"] > 0.05)
    ]

    if sig.empty:
        parts.append(
            "No features show a statistically significant "
            f"Pearson correlation with '{target_name}'."
        )
    else:
        top_feat = sig["pearson_r"].abs().idxmax()
        r_val = float(cont.loc[top_feat, "pearson_r"])
        p_val = float(cont.loc[top_feat, "pearson_p"])
        direction = "positive" if r_val > 0 else "negative"
        strength = (
            "strong" if abs(r_val) >= 0.5
            else "moderate" if abs(r_val) >= 0.3
            else "weak"
        )
        parts.append(
            f"Pearson correlation found '{top_feat}' has "
            f"a {strength} {direction} linear association "
            f"with '{target_name}' "
            f"(r = {r_val:.3f}, p = {p_val:.3f})."
        )
        n_sig = len(sig)
        if n_sig > 1:
            parts.append(
                f"In total, {n_sig} features show a "
                f"significant Pearson correlation with "
                f"'{target_name}'."
            )

    if not no_linear.empty:
        nl_str = _join_names(no_linear.index.tolist())
        parts.append(
            f"{nl_str}: no significant linear correlation "
            f"detected with '{target_name}' "
            "(|r| < 0.1, p > 0.05)."
        )


def _spearman_narrative(
    result: pd.DataFrame,
    target_name: str,
    parts: list[str],
) -> None:
    """Append Spearman correlation sentences to *parts*.

    Rows where ``spearman_r`` is NaN (categorical features assessed via
    ANOVA) are silently skipped.
    """
    cont = result[result["spearman_r"].notna()]
    if cont.empty:
        return
    sig = cont[cont["spearman_significant"].astype(bool)]

    if sig.empty:
        parts.append(
            "No features show a statistically significant "
            f"Spearman correlation with '{target_name}'."
        )
    else:
        top_feat = sig["spearman_r"].abs().idxmax()
        r_val = float(cont.loc[top_feat, "spearman_r"])
        p_val = float(cont.loc[top_feat, "spearman_p"])
        direction = "positive" if r_val > 0 else "negative"
        strength = (
            "strong" if abs(r_val) >= 0.5
            else "moderate" if abs(r_val) >= 0.3
            else "weak"
        )
        parts.append(
            f"Spearman correlation found '{top_feat}' has "
            f"a {strength} {direction} monotonic "
            f"association with '{target_name}' "
            f"(rho = {r_val:.3f}, p = {p_val:.3f})."
        )
        n_sig = len(sig)
        if n_sig > 1:
            parts.append(
                f"In total, {n_sig} features show a "
                f"significant Spearman correlation with "
                f"'{target_name}'."
            )


def _anova_narrative(
    result: pd.DataFrame,
    target_name: str,
    parts: list[str],
) -> None:
    """Append ANOVA / eta-squared sentences for categorical features."""
    cat = result[result["eta_squared"].notna()]
    if cat.empty:
        return

    sig = cat[cat["anova_significant"].astype(bool)]
    non_sig = cat[~cat["anova_significant"].astype(bool)]

    if sig.empty:
        feats = _join_names(cat.index.tolist())
        parts.append(
            f"One-way ANOVA found no statistically significant "
            f"difference in '{target_name}' across groups of "
            f"{feats}."
        )
    else:
        top_feat = sig["eta_squared"].idxmax()
        eta2 = float(cat.loc[top_feat, "eta_squared"])
        p_val = float(cat.loc[top_feat, "anova_p"])
        strength = (
            "large" if eta2 >= 0.14
            else "medium" if eta2 >= 0.06
            else "small"
        )
        parts.append(
            f"One-way ANOVA found '{top_feat}' groups differ "
            f"significantly in '{target_name}' "
            f"(η² = {eta2:.3f}, p = {p_val:.3f}; {strength} effect)."
        )
        if len(sig) > 1:
            others = [f for f in sig.index if f != top_feat]
            parts.append(
                f"{_join_names(others)} also show a "
                f"significant group effect on '{target_name}'."
            )

    if not non_sig.empty:
        ns_str = _join_names(non_sig.index.tolist())
        parts.append(
            f"{ns_str}: ANOVA found no significant group "
            f"difference in '{target_name}' (p > 0.05)."
        )


def _label_groups(
    consensus_df: pd.DataFrame,
    parts: list[str],
) -> None:
    """Append importance-label group sentences to *parts*."""
    strong = consensus_df[
        consensus_df["importance_label"] == "strongly important"
    ]["feature"].tolist()
    moderate = consensus_df[
        consensus_df["importance_label"] == "moderately important"
    ]["feature"].tolist()
    weak = consensus_df[
        consensus_df["importance_label"]
        == "weakly important or inconsistent"
    ]["feature"].tolist()

    if strong:
        s_str = _join_names(strong)
        verb = "is" if len(strong) == 1 else "are"
        parts.append(
            f"{s_str} {verb} 'strongly important' "
            "(ranked in top-K by >= 75% of methods)."
        )

    if moderate:
        m_str = _join_names(moderate)
        verb = "is" if len(moderate) == 1 else "are"
        parts.append(
            f"{m_str} {verb} 'moderately important' "
            "(ranked in top-K by 50-74% of methods)."
        )

    if weak:
        w_str = _join_names(weak)
        verb = "is" if len(weak) == 1 else "are"
        parts.append(
            f"{w_str} {verb} 'weakly important or "
            "inconsistent across methods'."
        )


def _pca_corr_flag(
    consensus_df: pd.DataFrame,
    parts: list[str],
) -> None:
    """Flag features with high PCA rank but low correlation rank."""
    pca_col = "pca_rank"
    corr_col = None
    for candidate in [
        "pearson_rank",
        "spearman_rank",
        "correlation_rank",
    ]:
        if candidate in consensus_df.columns:
            corr_col = candidate
            break

    if pca_col not in consensus_df.columns or corr_col is None:
        return

    n_feats = len(consensus_df)
    top_k = max(3, n_feats // 3)
    flagged = consensus_df[
        (consensus_df[pca_col] <= top_k)
        & (consensus_df[corr_col] > n_feats - top_k)
    ]["feature"].tolist()

    if flagged:
        f_str = _join_names(flagged)
        parts.append(
            f"{f_str} rank highly in PCA but show low "
            "direct correlation — they may be important "
            "in combination with other features "
            "(nonlinear/interaction effect)."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interpret_lasso(result: pd.DataFrame, target_name: str) -> str:
    """Generate a plain-English summary of LASSO results.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nextscreen.features.lasso.run_lasso`, containing
        columns ``['feature', 'coefficient', 'abs_coefficient', 'rank']``.
    target_name : str
        Human-readable name of the target variable being analyzed.

    Returns
    -------
    str
        One-paragraph plain-English interpretation, ending with the standard
        domain-expertise caveat.
    """
    parts: list[str] = []
    nonzero = result[result["coefficient"] != 0.0]
    zero_feats = result.loc[
        result["coefficient"] == 0.0, "feature"
    ].tolist()

    if nonzero.empty:
        parts.append(
            f"LASSO assigned zero weight to all features "
            f"for '{target_name}', indicating no detectable "
            "linear relationships at the selected "
            "regularization strength."
        )
    else:
        top = nonzero.iloc[0]
        feat = str(top["feature"])
        coef = float(top["coefficient"])
        parts.append(
            f"LASSO identified '{feat}' as the most "
            f"influential feature for '{target_name}' "
            f"(standardized coefficient {coef:+.4f})."
        )
        if len(nonzero) > 1:
            rest = nonzero.iloc[1:]["feature"].tolist()
            parts.append(
                "Other features retained by LASSO: "
                f"{_join_names(rest)}."
            )

    if zero_feats:
        z_str = _join_names(zero_feats)
        pronoun = "it" if len(zero_feats) == 1 else "they"
        parts.append(
            f"{z_str} received zero weight from LASSO, "
            f"suggesting {pronoun} may be irrelevant "
            "under linear assumptions."
        )

    parts.append(_CAVEAT)
    return " ".join(parts)


def interpret_random_forest(
    result: pd.DataFrame, target_name: str
) -> str:
    """Generate a plain-English summary of Random Forest importance results.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nextscreen.features.random_forest.run_random_forest`.
    target_name : str
        Human-readable name of the target variable.

    Returns
    -------
    str
        One-paragraph plain-English interpretation.
    """
    parts: list[str] = []
    top = result.iloc[0]
    feat = str(top["feature"])
    imp = float(top["importance"])
    parts.append(
        f"Random Forest ranked '{feat}' as the most "
        f"important feature for '{target_name}' "
        f"(importance score {imp:.4f})."
    )

    if len(result) > 1:
        others = result.iloc[1:]["feature"].tolist()
        parts.append(
            "Other features in descending order of "
            f"importance: {_join_names(others)}."
        )

    parts.append(_CAVEAT)
    return " ".join(parts)


def interpret_shap(
    result: dict[str, object], target_name: str
) -> str:
    """Generate a plain-English summary of SHAP importance results.

    Parameters
    ----------
    result : dict
        Output of :func:`nextscreen.features.shap_analysis.run_shap`.
    target_name : str
        Human-readable name of the target variable.

    Returns
    -------
    str
        One-paragraph plain-English interpretation.
    """
    fi = result["feature_importance"]
    parts: list[str] = []
    top = fi.iloc[0]  # type: ignore[union-attr]
    feat = str(top["feature"])
    shap_val = float(top["mean_abs_shap"])
    parts.append(
        f"SHAP analysis identified '{feat}' as the most "
        f"impactful feature for '{target_name}' "
        f"(mean |SHAP| = {shap_val:.4f})."
    )

    n_feat = len(fi)  # type: ignore[arg-type]
    if n_feat > 1:
        others = fi.iloc[1:]["feature"].tolist()  # type: ignore[union-attr]
        parts.append(
            "Features of secondary SHAP importance "
            f"(descending): {_join_names(others)}."
        )

    parts.append(_CAVEAT)
    return " ".join(parts)


def interpret_pca(result: dict[str, object]) -> str:
    """Generate a plain-English summary of PCA results.

    Parameters
    ----------
    result : dict
        Output of :func:`nextscreen.features.pca.run_pca`.

    Returns
    -------
    str
        One-paragraph plain-English interpretation.
    """
    n = int(result["n_components"])  # type: ignore[arg-type]
    cum_var_arr = result["cumulative_variance"]
    cum_var = float(cum_var_arr[-1])  # type: ignore[index]
    feature_rank = result["feature_rank"]
    loadings = result["loadings"]
    parts: list[str] = []

    pct = f"{cum_var * 100:.1f}%"
    parts.append(
        f"PCA selected {n} component(s), together "
        f"explaining {pct} of the total feature-space "
        "variance."
    )

    pc_cols = loadings.columns.tolist()  # type: ignore[union-attr]
    for pc in pc_cols:
        col_abs = loadings[pc].abs()  # type: ignore[index]
        top_idx = col_abs.idxmax()
        top_load = float(
            loadings.loc[top_idx, pc]  # type: ignore[index]
        )
        parts.append(
            f"{pc} is most strongly associated with "
            f"'{top_idx}' (loading {top_load:+.3f})."
        )

    top_row = feature_rank.iloc[0]  # type: ignore[union-attr]
    tfeat = str(top_row["feature"])
    tload = float(top_row["max_loading"])
    parts.append(
        f"Overall, '{tfeat}' shows the highest PCA "
        f"loading (max |loading| = {tload:.3f}) across "
        "all retained components."
    )

    parts.append(_CAVEAT)
    return " ".join(parts)


def interpret_correlations(
    result: pd.DataFrame, target_name: str
) -> str:
    """Generate a plain-English summary of correlation results.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nextscreen.features.correlations.run_correlations`.
    target_name : str
        Human-readable name of the target variable.

    Returns
    -------
    str
        One-paragraph plain-English interpretation.
    """
    parts: list[str] = []
    has_pearson = "pearson_r" in result.columns
    has_spearman = "spearman_r" in result.columns
    has_anova = "eta_squared" in result.columns

    if has_pearson:
        _pearson_narrative(result, target_name, parts)
    if has_spearman:
        _spearman_narrative(result, target_name, parts)
    if has_anova:
        _anova_narrative(result, target_name, parts)

    parts.append(_CAVEAT)
    return " ".join(parts)


def interpret_consensus(
    consensus_df: pd.DataFrame,
    target_name: str,
    n_methods: int,
) -> str:
    """Generate a plain-English consensus interpretation across all methods.

    Applies the following rule hierarchy:

    - Feature in top-K by >= 75% of methods → "strongly important"
    - Feature in top-K by 50–74% → "moderately important"
    - Feature in top-K by < 50% → "weakly important or inconsistent"

    Always appends: *"These are data-driven suggestions. Domain expertise
    should guide final decisions."*

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Output of :func:`nextscreen.features.consensus.compute_consensus`
        after :func:`nextscreen.features.consensus.label_importance`.
    target_name : str
        Human-readable name of the target variable.
    n_methods : int
        Total number of methods that contributed to the consensus.

    Returns
    -------
    str
        Multi-sentence plain-English interpretation with per-feature
        commentary and a closing caveat.
    """
    parts: list[str] = []
    top = consensus_df.iloc[0]
    top_feat = str(top["feature"])
    n_top_k = int(top["n_methods_top_k"])
    parts.append(
        f"Across {n_methods} method(s), '{top_feat}' is "
        f"the highest-ranked feature for '{target_name}', "
        f"appearing in the top tier for "
        f"{n_top_k} of {n_methods} method(s)."
    )

    _label_groups(consensus_df, parts)
    _pca_corr_flag(consensus_df, parts)

    parts.append(_CAVEAT)
    return " ".join(parts)


def interpret_ard_gp(
    result: pd.DataFrame, target_name: str
) -> str:
    """Generate a plain-English summary of ARD-GP importance results.

    Parameters
    ----------
    result : pd.DataFrame
        Output of :func:`nextscreen.features.ard_gp.run_ard_gp`.
    target_name : str
        Human-readable name of the target variable.

    Returns
    -------
    str
        One-paragraph plain-English interpretation.
    """
    parts: list[str] = []
    top = result.iloc[0]
    feat = str(top["feature"])
    ls_val = float(top["lengthscale"])
    parts.append(
        f"The ARD Gaussian Process identified '{feat}' as the most "
        f"influential feature for '{target_name}', with the shortest "
        f"ARD lengthscale ({ls_val:.4f}). A short lengthscale means "
        "the model predicts rapid change in the response along that "
        "dimension — a strong signal of relevance."
    )

    if len(result) > 1:
        long_ls = result[
            result["lengthscale"] > result["lengthscale"].median() * 2
        ]["feature"].tolist()
        if long_ls:
            parts.append(
                f"{_join_names(long_ls)} "
                + ("have" if len(long_ls) > 1 else "has")
                + " long lengthscales, suggesting the GP response "
                "surface is relatively flat in "
                + ("those directions" if len(long_ls) > 1
                   else "that direction")
                + " and "
                + ("those features" if len(long_ls) > 1
                   else "that feature")
                + " may be less critical."
            )
        others = result.iloc[1:]["feature"].tolist()
        parts.append(
            "Remaining features in descending order of "
            f"GP importance: {_join_names(others)}."
        )

    parts.append(_CAVEAT)
    return " ".join(parts)
