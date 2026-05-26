"""NEXTscreen Streamlit application entry point.

Run with::

    nextscreen

or directly via::

    streamlit run nextscreen/app.py
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nextscreen.data.loader import (
    detect_replicates,
    encode_categoricals,
    handle_replicates,
)
from nextscreen.features.consensus import (
    compute_consensus,
    label_importance,
)
from nextscreen.features.correlations import run_correlations
from nextscreen.features.lasso import run_lasso
from nextscreen.features.pca import run_pca
from nextscreen.features.random_forest import run_random_forest
from nextscreen.features.shap_analysis import run_shap
from nextscreen.interpretation.narrator import (
    interpret_consensus,
    interpret_correlations,
    interpret_lasso,
    interpret_pca,
    interpret_random_forest,
    interpret_shap,
)
from nextscreen.nextorch_integration.handoff import (
    build_parameter_space,
    make_scalarized_target,
    run_optimization,
    run_pareto_optimization,
)
from nextscreen.reporting.report import (
    build_html_report,
    export_pdf,
)
from nextscreen.utils.plotting import (
    bar_chart,
    correlation_heatmap,
    pareto_front_plot,
    pca_loading_heatmap,
    pca_variance_plot,
    shap_beeswarm,
)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NEXTscreen",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, object] = {
    "step": 1,
    "raw_df": None,
    "file_name": "",
    "feature_cols": [],
    "target_cols": [],
    "processed_df": None,
    "replicate_summary": None,
    "replicate_strategy": "average",
    "selected_methods": [
        "lasso", "random_forest", "shap", "pca", "correlations",
    ],
    "method_params": {},
    "feature_results": {},
    "consensus_results": {},
    "interpretations": {},
    "selected_features": [],
    "bounds": {},
    "suggested_experiments": None,
    "report_path": None,
    "categorical_maps": {},
    "fixed_conditions": {},
    "run_bootstrap": False,
    "n_bootstrap": 100,
    "mo_suggestions": {},
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
STEPS = {
    1: "📂 Data Upload",
    2: "🔁 Replicate Handling",
    3: "⚙️ Correlation Analysis",
    4: "📊 Analysis Results",
    5: "✅ Choose Variables",
    6: "🚀 NEXTorch Optimization",
    7: "📄 Report Export",
}

_DATA_READY = st.session_state.raw_df is not None
_PROCESSED_READY = st.session_state.processed_df is not None
_RESULTS_READY = bool(st.session_state.feature_results)
_FEATURES_CHOSEN = bool(st.session_state.selected_features)

_STEP_UNLOCKED = {
    1: True,
    2: _DATA_READY,
    3: _PROCESSED_READY,
    4: _RESULTS_READY,
    5: _RESULTS_READY,
    6: _FEATURES_CHOSEN,
    7: _FEATURES_CHOSEN,
}

with st.sidebar:
    st.title("NEXTscreen")
    st.caption("Screen. Understand. Optimize.")
    st.divider()

    for step_num, step_label in STEPS.items():
        is_current = st.session_state.step == step_num
        unlocked = _STEP_UNLOCKED[step_num]
        if st.button(
            step_label,
            key=f"nav_{step_num}",
            use_container_width=True,
            disabled=not unlocked,
            type="primary" if is_current else "secondary",
        ):
            st.session_state.step = step_num
            st.rerun()

    st.divider()
    st.caption(f"Step {st.session_state.step} of {len(STEPS)}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _step_header(title: str, description: str) -> None:
    st.title(title)
    st.markdown(description)
    st.divider()


def _advance(to_step: int) -> None:
    st.session_state.step = to_step
    st.rerun()


def _prep_rankings(
    results: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Extract (feature, rank) DataFrames from raw method results."""
    rankings: dict[str, pd.DataFrame] = {}
    for mname, res in results.items():
        if mname.startswith("_"):  # skip internal keys like _bootstrap
            continue
        if mname == "correlations":
            df = res[["rank"]].reset_index()  # type: ignore[union-attr]
            rankings[mname] = df
        elif mname == "shap":
            fi = res["feature_importance"]  # type: ignore[index]
            rankings[mname] = fi[["feature", "rank"]]
        elif mname == "pca":
            fr = res["feature_rank"]  # type: ignore[index]
            rankings[mname] = fr[["feature", "rank"]]
        else:
            rankings[mname] = res[["feature", "rank"]]  # type: ignore[index]
    return rankings


# ---------------------------------------------------------------------------
# Step 1 — Data Upload
# ---------------------------------------------------------------------------

def render_step1() -> None:
    _step_header(
        "📂 Step 1 — Data Upload",
        "Upload a CSV or Excel file containing your experimental data.",
    )

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV (.csv), Excel (.xlsx, .xls)",
    )

    if uploaded is not None:
        try:
            raw_bytes = uploaded.read()
            name = uploaded.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(raw_bytes))
            else:
                df = pd.read_excel(io.BytesIO(raw_bytes))

            if df.empty and df.columns.empty:
                st.error("The uploaded file appears to be empty.")
                return

            st.success(
                f"Loaded **{uploaded.name}** — "
                f"{len(df)} rows × {len(df.columns)} columns."
            )
            st.dataframe(df.head(10), use_container_width=True)

            all_cols = df.columns.tolist()

            target_cols = st.multiselect(
                "Select **target** column(s) — the variable(s) "
                "you are trying to predict or optimize:",
                options=all_cols,
                default=st.session_state.target_cols
                if st.session_state.target_cols
                else [],
                key="target_selector",
            )

            if target_cols:
                remaining = [
                    c for c in all_cols if c not in target_cols
                ]
                feature_cols = st.multiselect(
                    "Select **feature** (input) column(s):",
                    options=remaining,
                    default=remaining,
                    key="feature_selector",
                )

                if feature_cols:
                    selected_cols = feature_cols + target_cols
                    missing_counts = df[selected_cols].isnull().sum()
                    cols_with_missing = missing_counts[missing_counts > 0]
                    if not cols_with_missing.empty:
                        msg = ", ".join(
                            f"**{col}** ({n} missing)"
                            for col, n in cols_with_missing.items()
                        )
                        st.warning(
                            f"⚠️ The following selected columns contain "
                            f"missing values: {msg}. Rows with missing "
                            f"values in these columns will be dropped before "
                            f"analysis."
                        )

                    if st.button(
                        "Proceed to Step 2 →",
                        type="primary",
                    ):
                        st.session_state.raw_df = df
                        st.session_state.file_name = (
                            uploaded.name
                        )
                        st.session_state.target_cols = (
                            target_cols
                        )
                        st.session_state.feature_cols = (
                            feature_cols
                        )
                        # Reset downstream state.
                        st.session_state.processed_df = None
                        st.session_state.feature_results = {}
                        st.session_state.consensus_results = {}
                        st.session_state.selected_features = []
                        _advance(2)
                else:
                    st.warning(
                        "Please select at least one feature column."
                    )
            else:
                st.info(
                    "Select at least one target column to continue."
                )

        except Exception as exc:
            st.error(f"Could not load file: {exc}")

    elif st.session_state.raw_df is not None:
        st.info(
            f"Currently loaded: **{st.session_state.file_name}** "
            f"({len(st.session_state.raw_df)} rows). "
            "Re-upload to change the dataset."
        )
        if st.button("Proceed to Step 2 →", type="primary"):
            _advance(2)

    st.divider()
    with st.expander("🎲 Don't have data yet? Generate a Latin Hypercube sampling plan"):
        st.markdown(
            "Define your input variables and their operating ranges. "
            "NEXTscreen will generate a space-filling set of experiments "
            "for you to run. Once you have results, upload them above."
        )

        n_features = st.number_input(
            "Number of input variables",
            min_value=1, max_value=20, value=3,
            key="lhs_n_features",
        )

        lhs_default = pd.DataFrame({
            "Feature name": [f"X{i + 1}" for i in range(n_features)],
            "Lower bound": [0.0] * n_features,
            "Upper bound": [1.0] * n_features,
            "Step size (optional)": [None] * n_features,
        })
        lhs_table = st.data_editor(
            lhs_default,
            num_rows="fixed",
            use_container_width=True,
            key="lhs_table",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            n_points = st.number_input(
                "Number of experiments",
                min_value=2, max_value=500, value=10,
                key="lhs_n_points",
            )
        with col_b:
            lhs_seed = st.number_input(
                "Random seed", min_value=0, max_value=9999, value=42,
                key="lhs_seed",
            )

        if st.button("Generate LHS design", key="lhs_generate"):
            try:
                from nextorch import doe as nextorch_doe

                names = lhs_table["Feature name"].tolist()
                lowers = lhs_table["Lower bound"].to_numpy(dtype=float)
                uppers = lhs_table["Upper bound"].to_numpy(dtype=float)

                if any(lo >= hi for lo, hi in zip(lowers, uppers)):
                    st.error(
                        "Each lower bound must be strictly less than its "
                        "upper bound."
                    )
                else:
                    steps = lhs_table["Step size (optional)"].tolist()

                    X_norm = nextorch_doe.latin_hypercube(
                        n_dim=int(n_features),
                        n_points=int(n_points),
                        seed=int(lhs_seed),
                    )
                    X_real = lowers + X_norm * (uppers - lowers)

                    for j, step in enumerate(steps):
                        try:
                            s = float(step)
                            if s > 0:
                                X_real[:, j] = (
                                    np.round((X_real[:, j] - lowers[j]) / s)
                                    * s + lowers[j]
                                )
                        except (TypeError, ValueError):
                            pass

                    lhs_df = pd.DataFrame(X_real, columns=names).round(6)
                    st.dataframe(lhs_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download as CSV",
                        data=lhs_df.to_csv(index=False),
                        file_name="lhs_design.csv",
                        mime="text/csv",
                        key="lhs_download",
                    )
                    st.info(
                        "Run these experiments, add your measured response "
                        "as a new column, then upload the completed file above."
                    )
            except Exception as exc:
                st.error(f"Could not generate LHS design: {exc}")


# ---------------------------------------------------------------------------
# Step 2 — Replicate Handling
# ---------------------------------------------------------------------------

def render_step2() -> None:
    _step_header(
        "🔁 Step 2 — Replicate Handling",
        "Review detected replicates and choose how to handle them.",
    )

    if st.session_state.raw_df is None:
        st.warning("Please upload data in Step 1 first.")
        return

    df: pd.DataFrame = st.session_state.raw_df
    feature_cols: list[str] = st.session_state.feature_cols
    target_cols: list[str] = st.session_state.target_cols

    try:
        rep_df = detect_replicates(df, feature_cols)
    except Exception as exc:
        st.error(f"Replicate detection failed: {exc}")
        return

    if rep_df.empty:
        st.success(
            "No replicates detected — each combination of "
            "feature values is unique."
        )
    else:
        n_groups = rep_df["replicate_group"].nunique()
        st.warning(
            f"Detected **{len(rep_df)} replicate rows** "
            f"across **{n_groups} group(s)**."
        )
        st.dataframe(rep_df, use_container_width=True)

    strategy_labels = {
        "average": "Average replicates (recommended)",
        "keep_all": "Keep all rows unchanged",
        "std_uncertainty": (
            "Average + add std uncertainty column"
        ),
    }
    strategy = st.radio(
        "How should replicates be handled?",
        options=list(strategy_labels.keys()),
        format_func=lambda k: strategy_labels[k],
        index=list(strategy_labels.keys()).index(
            st.session_state.replicate_strategy
        ),
    )

    if st.button("Confirm and proceed to Step 3 →", type="primary"):
        try:
            processed, summary = handle_replicates(
                df,
                feature_cols,
                target_cols,
                strategy=strategy,  # type: ignore[arg-type]
            )
            st.session_state.replicate_summary = summary
            st.session_state.replicate_strategy = strategy

            # Auto-encode any string/object feature columns.
            _obj_feats = [
                c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(processed[c])
            ]
            if _obj_feats:
                processed, _cat_maps = encode_categoricals(
                    processed, cols=_obj_feats
                )
                st.session_state.categorical_maps = _cat_maps
            else:
                st.session_state.categorical_maps = {}

            st.session_state.processed_df = processed
            # Reset downstream.
            st.session_state.feature_results = {}
            st.session_state.consensus_results = {}
            st.session_state.selected_features = []
            _advance(3)
        except Exception as exc:
            st.error(f"Replicate handling failed: {exc}")


# ---------------------------------------------------------------------------
# Step 3 — Method Selection & Execution
# ---------------------------------------------------------------------------

def render_step3() -> None:
    _step_header(
        "⚙️ Step 3 — Correlation Analysis",
        "Choose analysis methods to understand which variables "
        "correlate with and drive your target(s).",
    )

    if st.session_state.processed_df is None:
        st.warning("Please complete Step 2 first.")
        return

    st.subheader("Analysis methods")
    col1, col2 = st.columns(2)
    with col1:
        use_lasso = st.checkbox("LASSO", 
                                value=True,
                                help=(
                                    "LASSO is a linear model that identifies features most strongly correlated with the target" 
                                    "while accounting for the other variables; it estimates feature importance based on the magnitude of each nonzero coefficient," 
                                    "where larger absolute coefficients indicate stronger positive or negative contributions to the prediction," 
                                    "and coefficients shrunk to zero are treated as unimportant or redundant features."
                                ))
        use_rf = st.checkbox("Random Forest", 
                             value=True,
                             help=(
                                 "Random Forest is an ensemble model that combines multiple decision “trees”" 
                                 "to form “forests” to make predictions; it estimates feature importance" 
                                 "based on how much each variable contributes to reducing prediction error" 
                                 "across the trees, capturing nonlinear effects"
                             )
                             )
        use_shap = st.checkbox("SHAP", value=True,
                               help=(
                                   "SHAP explains individual predictions by" 
                                   "showing how each variable pushed the prediction higher or lower relative to the average"
                               ))
    with col2:
        use_pca = st.checkbox("PCA", 
                              value=True,
                              help = ("PCA is an unsupervised dimensionality reduction method that identifies combinations of correlated input variables," 
                              "called principal components, that capture the largest directions of variance in the dataset;" 
                              "it estimates feature relevance based on how strongly each variable loads onto the main components," 
                              "showing which variables contribute most to the dominant patterns in the data rather than directly to the prediction target"))
        use_corr = st.checkbox(
            "Pearson / Spearman Correlations", value=True,
            help=(
                "Pearson coefficient represents linear relationship and Spearman coefficient represents monotonic relationship"
            )
        )
    run_bootstrap = st.checkbox(
        "Compute bootstrap confidence intervals on ranks",
        value=st.session_state.run_bootstrap,
        help=(
            "Resample training data and rerun each method "
            "multiple times to estimate rank stability. "
            "Slower but shows which rankings are reliable."
        ),
    )

    with st.expander("Advanced Settings", expanded=False):
        st.markdown("**LASSO**")
        lasso_alpha_auto = st.checkbox(
            "Auto-tune alpha via cross-validation (recommended)",
            value=True,
        )
        lasso_alpha = None
        if not lasso_alpha_auto:
            lasso_alpha = st.number_input(
                "Alpha", min_value=1e-6, value=0.01, format="%.6f"
            )

        st.markdown("**Random Forest**")
        rf_n_est = st.number_input(
            "n_estimators", min_value=1, value=100
        )
        rf_max_depth_raw = st.number_input(
            "max_depth (0 = unlimited)", min_value=0, value=0
        )
        rf_max_depth = int(rf_max_depth_raw) or None

        st.markdown("**PCA**")
        pca_thresh = st.slider(
            "Variance threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.90,
            step=0.05,
        )
        pca_max_comp = st.number_input(
            "Max components", min_value=1, value=5
        )

        st.markdown("**SHAP**")
        shap_bg = st.number_input(
            "Background samples", min_value=1, value=100
        )

        st.markdown("**Correlations**")
        corr_method = st.selectbox(
            "Method",
            options=["both", "pearson", "spearman"],
            index=0,
        )
        corr_sig = st.number_input(
            "Significance threshold (p)",
            min_value=0.001,
            max_value=0.20,
            value=0.05,
            step=0.005,
            format="%.3f",
        )

        st.markdown("**Bootstrap**")
        n_bootstrap = st.number_input(
            "Bootstrap samples",
            min_value=20,
            max_value=500,
            value=int(st.session_state.n_bootstrap),
        )

    any_selected = any(
        [use_lasso, use_rf, use_shap, use_pca, use_corr]
    )
    if not any_selected:
        st.warning("Select at least one method to continue.")
        return

    if st.button("▶ Run analysis", type="primary"):
        st.session_state.run_bootstrap = run_bootstrap
        st.session_state.n_bootstrap = int(n_bootstrap)
        _run_feature_selection(
            use_lasso=use_lasso,
            use_rf=use_rf,
            use_shap=use_shap,
            use_pca=use_pca,
            use_corr=use_corr,
            run_bootstrap=run_bootstrap,
            lasso_alpha=lasso_alpha,
            rf_n_est=int(rf_n_est),
            rf_max_depth=rf_max_depth,
            pca_thresh=float(pca_thresh),
            pca_max_comp=int(pca_max_comp),
            shap_bg=int(shap_bg),
            corr_method=str(corr_method),
            corr_sig=float(corr_sig),
            n_bootstrap=int(n_bootstrap),
        )


def _run_feature_selection(
    *,
    use_lasso: bool,
    use_rf: bool,
    use_shap: bool,
    use_pca: bool,
    use_corr: bool,
    run_bootstrap: bool,
    lasso_alpha: float | None,
    rf_n_est: int,
    rf_max_depth: int | None,
    pca_thresh: float,
    pca_max_comp: int,
    shap_bg: int,
    corr_method: str,
    corr_sig: float,
    n_bootstrap: int,
) -> None:
    """Run all selected methods and store results in session state."""
    processed: pd.DataFrame = st.session_state.processed_df
    feature_cols: list[str] = st.session_state.feature_cols
    target_cols: list[str] = st.session_state.target_cols
    X = processed[feature_cols]

    feature_results: dict[str, dict[str, object]] = {}
    consensus_results: dict[str, object] = {}

    progress = st.progress(0, text="Running feature selection…")
    n_targets = len(target_cols)

    for t_idx, target in enumerate(target_cols):
        y = processed[target]
        results: dict[str, object] = {}
        methods_done = 0
        n_methods = sum(
            [use_lasso, use_rf, use_shap, use_pca, use_corr]
        )

        try:
            if use_lasso:
                progress.progress(
                    _prog(t_idx, n_targets, methods_done, n_methods),
                    text=f"[{target}] Running LASSO…",
                )
                results["lasso"] = run_lasso(
                    X, y, alpha=lasso_alpha
                )
                methods_done += 1

            if use_rf:
                progress.progress(
                    _prog(t_idx, n_targets, methods_done, n_methods),
                    text=f"[{target}] Running Random Forest…",
                )
                results["random_forest"] = run_random_forest(
                    X, y,
                    n_estimators=rf_n_est,
                    max_depth=rf_max_depth,
                )
                methods_done += 1

            if use_shap:
                progress.progress(
                    _prog(t_idx, n_targets, methods_done, n_methods),
                    text=f"[{target}] Running SHAP…",
                )
                results["shap"] = run_shap(
                    X, y, background_samples=shap_bg
                )
                methods_done += 1

            if use_pca:
                progress.progress(
                    _prog(t_idx, n_targets, methods_done, n_methods),
                    text=f"[{target}] Running PCA…",
                )
                results["pca"] = run_pca(
                    X,
                    variance_threshold=pca_thresh,
                    max_components=pca_max_comp,
                )
                methods_done += 1

            if use_corr:
                progress.progress(
                    _prog(t_idx, n_targets, methods_done, n_methods),
                    text=f"[{target}] Running Correlations…",
                )
                results["correlations"] = run_correlations(
                    X, y,
                    method=corr_method,  # type: ignore[arg-type]
                    significance_threshold=corr_sig,
                    categorical_cols=list(
                        st.session_state.get(
                            "categorical_maps", {}
                        ).keys()
                    ),
                )
                methods_done += 1

        except Exception as exc:
            st.error(
                f"Error during feature selection "
                f"for '{target}': {exc}"
            )
            progress.empty()
            return

        feature_results[target] = results

        # Bootstrap CI (optional).
        if run_bootstrap:
            from nextscreen.features.bootstrap import (  # noqa: PLC0415
                bootstrap_ranks,
                extract_correlations,
                extract_pca,
                extract_shap,
                extract_tabular,
            )
            _method_extractors = {
                "lasso": (run_lasso, extract_tabular,
                          {"alpha": lasso_alpha}),
                "random_forest": (
                    run_random_forest, extract_tabular,
                    {"n_estimators": rf_n_est,
                     "max_depth": rf_max_depth}
                ),
                "shap": (run_shap, extract_shap,
                         {"background_samples": shap_bg}),
                "correlations": (
                    run_correlations, extract_correlations,
                    {"method": corr_method,
                     "significance_threshold": corr_sig,
                     "categorical_cols": list(
                         st.session_state.get(
                             "categorical_maps", {}
                         ).keys()
                     )}
                ),
            }
            bootstrap_results: dict[str, pd.DataFrame] = {}
            for mname, (mfn, extractor, mkwargs) in (
                _method_extractors.items()
            ):
                if mname not in results:
                    continue
                progress.progress(
                    _prog(t_idx, n_targets,
                          methods_done, n_methods),
                    text=(
                        f"[{target}] Bootstrap {mname} "
                        f"({n_bootstrap} samples)…"
                    ),
                )
                try:
                    bootstrap_results[mname] = (
                        bootstrap_ranks(
                            method=mfn,
                            X=X,
                            y=y,
                            result_extractor=extractor,
                            n_bootstrap=n_bootstrap,
                            method_kwargs=mkwargs,
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
            # PCA uses X only.
            if "pca" in results:
                try:
                    bootstrap_results["pca"] = (
                        bootstrap_ranks(
                            method=run_pca,
                            X=X,
                            y=None,
                            result_extractor=extract_pca,
                            n_bootstrap=n_bootstrap,
                            method_kwargs={
                                "variance_threshold": pca_thresh,
                                "max_components": pca_max_comp,
                            },
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
            feature_results[target]["_bootstrap"] = (
                bootstrap_results
            )

        # Build consensus.
        try:
            rankings = _prep_rankings(results)
            con_df = compute_consensus(rankings)
            con_df = label_importance(con_df, n_methods=len(rankings))
            consensus_results[target] = con_df
        except Exception as exc:
            st.error(
                f"Consensus ranking failed for '{target}': {exc}"
            )
            progress.empty()
            return

    progress.progress(1.0, text="Done!")
    progress.empty()

    st.session_state.feature_results = feature_results
    st.session_state.consensus_results = consensus_results
    st.success(
        f"Analysis complete for "
        f"{len(target_cols)} target(s). "
        "Proceeding to results…"
    )
    _advance(4)


def _prog(
    t_idx: int, n_targets: int, m_done: int, n_methods: int
) -> float:
    """Compute a 0–1 progress fraction."""
    base = t_idx / n_targets
    within = (m_done / n_methods) / n_targets
    return min(base + within, 0.99)


# ---------------------------------------------------------------------------
# Step 4 — Feature Selection Results
# ---------------------------------------------------------------------------

def render_step4() -> None:
    _step_header(
        "📊 Step 4 — Analysis Results",
        "Explore variable importance scores, correlation charts, "
        "and the consensus ranking across methods.",
    )

    if not st.session_state.feature_results:
        st.warning("Please run the analysis in Step 3 first.")
        return

    target_cols: list[str] = st.session_state.target_cols
    feature_results: dict = st.session_state.feature_results
    consensus_results: dict = st.session_state.consensus_results

    # One outer tab per target, then method tabs inside.
    if len(target_cols) == 1:
        _render_target_results(
            target_cols[0],
            feature_results[target_cols[0]],
            consensus_results.get(target_cols[0]),
        )
    else:
        outer_tabs = st.tabs(target_cols)
        for tab, target in zip(outer_tabs, target_cols):
            with tab:
                _render_target_results(
                    target,
                    feature_results.get(target, {}),
                    consensus_results.get(target),
                )

    if st.button("Proceed to Step 5 →", type="primary"):
        _advance(5)


def _render_target_results(
    target: str,
    results: dict[str, object],
    consensus_df: pd.DataFrame | None,
) -> None:
    """Render all method tabs for a single target."""
    bootstrap_map: dict = (
        results.pop("_bootstrap", {}) or {}
    )
    method_names = list(results.keys())
    tab_labels = [_method_label(m) for m in method_names]
    if consensus_df is not None:
        tab_labels.append("📋 Consensus")

    tabs = st.tabs(tab_labels)

    for tab, mname in zip(tabs, method_names):
        with tab:
            _render_method_tab(
                mname, results[mname], target,
                bootstrap_ci=bootstrap_map.get(mname),
            )

    if consensus_df is not None:
        with tabs[-1]:
            _render_consensus_tab(
                consensus_df, target, len(results)
            )


def _method_label(name: str) -> str:
    return {
        "lasso": "🔵 LASSO",
        "random_forest": "🌲 Random Forest",
        "shap": "🔮 SHAP",
        "pca": "📐 PCA",
        "correlations": "📈 Correlations",
    }.get(name, name.title())


def _render_method_tab(
    mname: str, result: object, target: str,
    bootstrap_ci: pd.DataFrame | None = None,
) -> None:
    if mname == "lasso":
        _tab_lasso(result, target)  # type: ignore[arg-type]
    elif mname == "random_forest":
        _tab_rf(result, target)  # type: ignore[arg-type]
    elif mname == "shap":
        _tab_shap(result, target)  # type: ignore[arg-type]
    elif mname == "pca":
        _tab_pca(result, target)  # type: ignore[arg-type]
    elif mname == "correlations":
        _tab_corr(result, target)  # type: ignore[arg-type]
    if bootstrap_ci is not None and not bootstrap_ci.empty:
        _render_bootstrap_ci(bootstrap_ci, mname)


def _tab_lasso(
    result: pd.DataFrame, target: str
) -> None:
    st.subheader(f"LASSO — {target}")
    col1, col2 = st.columns([3, 2])
    with col1:
        try:
            fig = bar_chart(
                result, "feature", "coefficient",
                f"LASSO Coefficients — {target}",
                x_label="Feature",
                y_label="Standardised coefficient",
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"lasso_{target}")
        except Exception as exc:
            st.warning(f"Chart unavailable: {exc}")
    with col2:
        st.dataframe(
            result[["feature", "coefficient", "rank"]],
            use_container_width=True,
            hide_index=True,
        )
    try:
        st.info(interpret_lasso(result, target))
    except Exception:
        pass


def _tab_rf(result: pd.DataFrame, target: str) -> None:
    st.subheader(f"Random Forest — {target}")
    col1, col2 = st.columns([3, 2])
    with col1:
        try:
            fig = bar_chart(
                result, "feature", "importance",
                f"RF Feature Importances — {target}",
                x_label="Feature",
                y_label="Importance",
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"rf_{target}")
        except Exception as exc:
            st.warning(f"Chart unavailable: {exc}")
    with col2:
        st.dataframe(
            result[["feature", "importance", "rank"]],
            use_container_width=True,
            hide_index=True,
        )
    try:
        st.info(interpret_random_forest(result, target))
    except Exception:
        pass


def _tab_shap(
    result: dict[str, object], target: str
) -> None:
    st.subheader(f"SHAP — {target}")
    fi: pd.DataFrame = result["feature_importance"]  # type: ignore
    col1, col2 = st.columns([3, 2])
    with col1:
        try:
            fig = bar_chart(
                fi, "feature", "mean_abs_shap",
                f"SHAP Feature Importances — {target}",
                x_label="Feature",
                y_label="Mean |SHAP|",
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"shap_bar_{target}")
        except Exception as exc:
            st.warning(f"Bar chart unavailable: {exc}")
    with col2:
        st.dataframe(
            fi[["feature", "mean_abs_shap", "rank"]],
            use_container_width=True,
            hide_index=True,
        )
    try:
        sv = result["shap_values"]
        X_bg = result["X_background"]
        fig2 = shap_beeswarm(
            sv, X_bg,  # type: ignore
            title=f"SHAP Beeswarm — {target}",
        )
        st.plotly_chart(fig2, use_container_width=True,
                        key=f"shap_beeswarm_{target}")
    except Exception as exc:
        st.warning(f"Beeswarm unavailable: {exc}")
    try:
        st.info(interpret_shap(result, target))
    except Exception:
        pass


def _tab_pca(
    result: dict[str, object], target: str
) -> None:
    st.subheader(f"PCA — {target}")
    col1, col2 = st.columns(2)
    with col1:
        try:
            fig = pca_variance_plot(
                result["explained_variance_ratio"],  # type: ignore
                title="PCA Explained Variance",
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"pca_variance_{target}")
        except Exception as exc:
            st.warning(f"Variance plot unavailable: {exc}")
    with col2:
        try:
            fig2 = pca_loading_heatmap(
                result["loadings"],  # type: ignore
                title="PCA Component Loadings",
            )
            st.plotly_chart(fig2, use_container_width=True,
                            key=f"pca_loading_{target}")
        except Exception as exc:
            st.warning(f"Loading heatmap unavailable: {exc}")
    _cum_arr = result["cumulative_variance"]  # type: ignore[index]
    _cum_pct = float(_cum_arr[-1]) * 100  # type: ignore[index]
    st.caption(
        f"Components selected: {result['n_components']} — "
        f"cumulative variance: {_cum_pct:.1f}%"
    )
    try:
        st.info(interpret_pca(result))
    except Exception:
        pass


def _tab_corr(
    result: pd.DataFrame, target: str
) -> None:
    st.subheader(f"Correlations — {target}")
    heatmap_data: dict[str, object] = {}
    if "pearson_r" in result.columns:
        heatmap_data["Pearson r"] = result["pearson_r"]
    if "spearman_r" in result.columns:
        heatmap_data["Spearman rho"] = result["spearman_r"]

    if heatmap_data:
        try:
            hm_df = pd.DataFrame(heatmap_data)
            fig = correlation_heatmap(
                hm_df,
                title=f"Feature\u2013Target Correlations — {target}",
            )
            col1, col2 = st.columns([3, 2])
            with col1:
                st.plotly_chart(
                    fig, use_container_width=True,
                    key=f"corr_heatmap_{target}",
                )
            with col2:
                display_cols = [
                    c for c in result.columns
                    if c in (
                        "pearson_r", "pearson_p",
                        "pearson_significant",
                        "spearman_r", "spearman_p",
                        "spearman_significant", "rank",
                    )
                ]
                st.dataframe(
                    result[display_cols].reset_index(),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as exc:
            st.warning(f"Chart unavailable: {exc}")
    try:
        st.info(interpret_correlations(result, target))
    except Exception:
        pass




def _render_bootstrap_ci(
    ci_df: pd.DataFrame, method_name: str
) -> None:
    with st.expander(
        f"📊 Bootstrap CI — {_method_label(method_name)}"
    ):
        st.caption(
            "Ranks recomputed across bootstrap resamples. "
            "Narrow IQR = stable rank; wide IQR = uncertain."
        )
        st.dataframe(
            ci_df[["feature", "median_rank",
                   "rank_q25", "rank_q75", "rank_std"]],
            use_container_width=True,
            hide_index=True,
        )


def _render_consensus_tab(
    consensus_df: pd.DataFrame,
    target: str,
    n_methods: int,
) -> None:
    st.subheader(f"Consensus Ranking — {target}")
    col1, col2 = st.columns([3, 2])
    with col1:
        try:
            fig = bar_chart(
                consensus_df,
                "feature", "n_methods_top_k",
                f"Consensus — {target}",
                x_label="Feature",
                y_label="# Methods ranking in top-K",
                color_col="importance_label",
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"consensus_{target}")
        except Exception as exc:
            st.warning(f"Consensus chart unavailable: {exc}")
    with col2:
        disp_cols = [
            c for c in [
                "feature", "consensus_rank",
                "n_methods_top_k", "importance_label",
                "avg_normalized_rank",
            ]
            if c in consensus_df.columns
        ]
        st.dataframe(
            consensus_df[disp_cols],
            use_container_width=True,
            hide_index=True,
        )
    try:
        st.info(
            interpret_consensus(consensus_df, target, n_methods)
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Step 5 — Feature Decision
# ---------------------------------------------------------------------------

def render_step5() -> None:
    _step_header(
        "✅ Step 5 — Choose Variables to Optimize",
        "The analysis above shows which variables appear most "
        "influential. Select the conditions you want to optimize — "
        "you are free to include all of them.",
    )

    if not st.session_state.consensus_results:
        st.warning("Please complete the analysis in Step 3 first.")
        return

    target_cols: list[str] = st.session_state.target_cols
    consensus_results: dict = st.session_state.consensus_results

    show_cross = False
    if len(target_cols) > 1:
        show_cross = st.toggle(
            "Show cross-target comparison", value=False
        )

    if show_cross and len(target_cols) > 1:
        _render_cross_target_table(
            target_cols, consensus_results
        )
    else:
        # Use first target for pre-selection.
        first_target = target_cols[0]
        con_df: pd.DataFrame = consensus_results[first_target]
        feature_cols: list[str] = st.session_state.feature_cols

        # Pre-check top-K features (strongly + moderately important).
        default_selected = con_df.loc[
            con_df["importance_label"].isin(
                ["strongly important", "moderately important"]
            ),
            "feature",
        ].tolist()
        if not default_selected:
            default_selected = (
                con_df["feature"].tolist()[:3]
            )

        st.markdown(
            "Variables highlighted by the analysis are pre-checked "
            "below. You can select or deselect any of them — "
            "the final choice is yours."
        )
        selected: list[str] = []
        for feat in feature_cols:
            label_row = con_df[con_df["feature"] == feat]
            label = (
                label_row["importance_label"].iloc[0]
                if not label_row.empty
                else "—"
            )
            rank_val = (
                int(label_row["consensus_rank"].iloc[0])
                if not label_row.empty
                else "—"
            )
            checked = st.checkbox(
                f"**{feat}** — rank {rank_val} | {label}",
                value=feat in default_selected,
                key=f"feat_check_{feat}",
            )
            if checked:
                selected.append(feat)

        if not selected:
            st.warning(
                "Select at least one feature to continue."
            )
            return

        if st.button(
            f"Confirm {len(selected)} variable(s) and "
            "proceed to Step 6 →",
            type="primary",
        ):
            st.session_state.selected_features = selected
            _cat_maps = st.session_state.get(
                "categorical_maps", {}
            )
            st.session_state.bounds = {
                f: (
                    {
                        "lower": float(
                            min(_cat_maps[f].values())
                        ),
                        "upper": float(
                            max(_cat_maps[f].values())
                        ),
                        "type": "categorical",
                        "values": sorted(
                            float(v)
                            for v in _cat_maps[f].values()
                        ),
                    }
                    if f in _cat_maps
                    else {
                        "lower": 0.0,
                        "upper": 1.0,
                        "type": "continuous",
                        "values": None,
                    }
                )
                for f in selected
            }
            # Compute fixed conditions: unselected variables
            # fixed at their best-observed value.
            _proc = st.session_state.processed_df
            _first_target = st.session_state.target_cols[0]
            _best_row = _proc.loc[
                _proc[_first_target].idxmax()
            ]
            _fixed: dict[str, float] = {}
            for _f in feature_cols:
                if _f not in selected:
                    _fixed[_f] = float(_best_row[_f])
            st.session_state.fixed_conditions = _fixed
            _advance(6)


def _render_cross_target_table(
    target_cols: list[str],
    consensus_results: dict,
) -> None:
    """Show a table comparing importance labels across targets."""
    st.subheader("Cross-target comparison")
    all_features: list[str] = st.session_state.feature_cols
    rows: list[dict] = []
    for feat in all_features:
        row: dict[str, str] = {"feature": feat}
        for tgt in target_cols:
            con_df = consensus_results.get(tgt)
            if con_df is not None:
                match = con_df[con_df["feature"] == feat]
                row[tgt] = (
                    match["importance_label"].iloc[0]
                    if not match.empty
                    else "—"
                )
            else:
                row[tgt] = "—"
        rows.append(row)
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Step 6 — NEXTorch Integration
# ---------------------------------------------------------------------------

def render_step6() -> None:
    _step_header(
        "🚀 Step 6 — Experiment Optimization",
        "Define variable bounds and generate suggested next "
        "experiments via Bayesian Optimization.",
    )

    if not st.session_state.selected_features:
        st.warning("Please select features in Step 5 first.")
        return

    selected: list[str] = st.session_state.selected_features
    bounds: dict = st.session_state.bounds

    _cat_maps: dict = st.session_state.get(
        "categorical_maps", {}
    )
    if _cat_maps:
        with st.expander(
            "🏷️ Auto-encoded categorical columns"
        ):
            for _col, _lmap in _cat_maps.items():
                _pairs = " | ".join(
                    f"`{k}` → {v}"
                    for k, v in sorted(
                        _lmap.items(),
                        key=lambda kv: kv[1],
                    )
                )
                st.markdown(f"**{_col}**: {_pairs}")

    # Display label → internal type string
    _TYPE_LABELS = {
        "Continuous range": "continuous",
        "Stepped range":    "integer",
        "Discrete values":  "categorical",
    }
    _TYPE_DISPLAY = {v: k for k, v in _TYPE_LABELS.items()}

    st.subheader("Variable bounds")
    for feat in selected:
        _default_b = {
            "lower": 0.0, "upper": 1.0,
            "type": "continuous", "values": None,
        }
        b = bounds.get(feat, _default_b)
        _cur_type = str(b.get("type", "continuous"))
        _cur_label = _TYPE_DISPLAY.get(
            _cur_type, "Continuous range"
        )
        c1, c2, c3 = st.columns([2, 2, 2])
        with c3:
            _chosen_label = st.selectbox(
                f"{feat} — type",
                options=list(_TYPE_LABELS.keys()),
                index=list(
                    _TYPE_LABELS.keys()
                ).index(_cur_label),
                key=f"vt_{feat}",
                help=(
                    "Continuous range: BO searches freely "
                    "within [lower, upper]. "
                    "Stepped range: BO picks values at fixed "
                    "increments (e.g. 100, 110, 120 … for step=10). "
                    "Discrete values: BO picks only from the "
                    "list you provide — use this for catalyst "
                    "types or other non-numeric options."
                ),
            )
            vtype = _TYPE_LABELS[_chosen_label]
        if vtype == "categorical":
            _saved = b.get("values") or [0.0, 1.0, 2.0]
            _vals_str = ", ".join(str(v) for v in _saved)
            with c1:
                raw_vals = st.text_input(
                    f"{feat} — discrete values",
                    value=_vals_str,
                    key=f"vals_{feat}",
                    help=(
                        "Comma-separated values the BO may "
                        "suggest. E.g. '100, 150, 200, 250, 300'"
                        " for temperature levels, or '0, 1, 2, 3'"
                        " for encoded catalyst types."
                    ),
                )
            with c2:
                st.caption("← comma-separated values")
            try:
                cat_vals = [
                    float(v.strip())
                    for v in raw_vals.split(",")
                    if v.strip()
                ]
                if len(cat_vals) < 2:
                    raise ValueError("need >= 2 values")
            except ValueError:
                st.warning(
                    f"'{feat}': enter at least 2 values "
                    "(e.g. '0, 1, 2'). Using defaults."
                )
                cat_vals = [0.0, 1.0, 2.0]
            bounds[feat] = {
                "lower": float(min(cat_vals)),
                "upper": float(max(cat_vals)),
                "type": "categorical",
                "values": cat_vals,
            }
        else:
            with c1:
                lower = st.number_input(
                    f"{feat} — lower",
                    value=float(b.get("lower", 0.0)),
                    key=f"lb_{feat}",
                )
            with c2:
                upper = st.number_input(
                    f"{feat} — upper",
                    value=float(b.get("upper", 1.0)),
                    key=f"ub_{feat}",
                )
            step = None
            if vtype == "integer":
                step = st.number_input(
                    f"{feat} — step size",
                    value=float(b.get("step") or 1.0),
                    min_value=1e-6,
                    key=f"step_{feat}",
                    help=(
                        "Increment between allowed values. "
                        "E.g. step=10 with lower=100, upper=200 "
                        "gives 100, 110, 120 … 200."
                    ),
                )
            bounds[feat] = {
                "lower": lower,
                "upper": upper,
                "type": vtype,
                "step": step,
                "values": None,
            }

    st.session_state.bounds = bounds

    # ── Fixed conditions ──────────────────────────────────────
    _fixed: dict = st.session_state.get("fixed_conditions", {})
    if _fixed:
        st.subheader("Fixed conditions")
        st.caption(
            "These variables are not being optimized. "
            "Values are pre-filled from your best observed "
            "experiment — adjust if needed."
        )
        _cat_maps_fix: dict = st.session_state.get(
            "categorical_maps", {}
        )
        _updated_fixed: dict[str, float] = {}
        for _fc, _fv in _fixed.items():
            if _fc in _cat_maps_fix:
                _decode = {
                    int(v): k
                    for k, v in _cat_maps_fix[_fc].items()
                }
                _label = _decode.get(int(round(_fv)), str(_fv))
                _opts = sorted(
                    _cat_maps_fix[_fc].items(),
                    key=lambda kv: kv[1],
                )
                _chosen = st.selectbox(
                    f"{_fc} (fixed)",
                    options=[k for k, _ in _opts],
                    index=(
                        [k for k, _ in _opts].index(_label)
                        if _label in [k for k, _ in _opts]
                        else 0
                    ),
                    key=f"fix_{_fc}",
                )
                _updated_fixed[_fc] = float(
                    _cat_maps_fix[_fc][_chosen]
                )
            else:
                _updated_fixed[_fc] = st.number_input(
                    f"{_fc} (fixed)",
                    value=_fv,
                    key=f"fix_{_fc}",
                )
        st.session_state.fixed_conditions = _updated_fixed
        _fixed = _updated_fixed

    target_cols: list[str] = (
        st.session_state.target_cols or []
    )
    if not target_cols:
        st.warning(
            "No target column available. Complete Step 1."
        )
        return

    # ── Helper: decode + display a suggestions DataFrame ──────
    def _show_suggestions(
        sugg_df: pd.DataFrame,
        download_name: str,
    ) -> None:
        _sugg = sugg_df.copy()
        _fixed_now: dict = st.session_state.get(
            "fixed_conditions", {}
        )
        _dec_maps: dict = st.session_state.get(
            "categorical_maps", {}
        )
        for _fc, _fv in _fixed_now.items():
            if _fc in _dec_maps:
                _dec = {
                    int(v): k
                    for k, v in _dec_maps[_fc].items()
                }
                _sugg[_fc] = _dec.get(
                    int(round(_fv)), str(_fv)
                )
            else:
                _sugg[_fc] = _fv
        for _col, _lmap in _dec_maps.items():
            if (
                _col in _sugg.columns
                and _col not in _fixed_now
            ):
                # Skip columns already decoded to string labels
                # (e.g. when categorical_maps was passed directly
                # to run_optimization / run_pareto_optimization).
                if pd.api.types.is_object_dtype(
                    _sugg[_col]
                ):
                    continue
                _dec = {
                    int(v): k for k, v in _lmap.items()
                }
                _sugg[_col] = (
                    pd.to_numeric(
                        _sugg[_col], errors="coerce"
                    )
                    .round()
                    .astype(int)
                    .map(_dec)
                )
        st.dataframe(_sugg, use_container_width=True)
        csv = _sugg.to_csv(index=False)
        st.download_button(
            "⬇ Download as CSV",
            data=csv,
            file_name=download_name,
            mime="text/csv",
        )
    st.number_input(
        "Number of suggested experiments",
        min_value=1,
        max_value=20,
        value=5,
        key="n_suggestions",
    )

    # ── Multi-objective toggle ─────────────────────────────────
    use_multi_obj = False
    if len(target_cols) > 1:
        use_multi_obj = st.toggle(
            "Multi-objective optimization",
            value=False,
            key="use_multi_obj",
            help=(
                "Optimize multiple targets simultaneously. "
                "Choose between weighted scalarization "
                "(explicit weight control) or Pareto-front "
                "BO (no weights, finds the full trade-off "
                "frontier automatically)."
            ),
        )

    if use_multi_obj:
        mo_strategy = st.radio(
            "Strategy",
            options=["Pareto front (qEHVI)", "Weighted scalarization"],
            key="mo_strategy",
            horizontal=True,
            help=(
                "**Pareto front**: fits one GP per target and uses "
                "q-Expected Hypervolume Improvement to suggest experiments "
                "that collectively push the Pareto front forward. No weights "
                "needed.\n\n"
                "**Weighted scalarization**: combines targets into a single "
                "score using user-controlled weights; runs separate BO for "
                "each scenario."
            ),
        )

        if mo_strategy == "Weighted scalarization":
            n_t = len(target_cols)
            _combos: list[tuple[str, list[float]]] = [
                (
                    f"Maximize {t}",
                    [1.0 if j == i else 0.0
                     for j in range(n_t)],
                )
                for i, t in enumerate(target_cols)
            ]
            _combos.append(
                ("Balanced", [1.0 / n_t] * n_t)
            )
            st.markdown(
                "**Optimization scenarios** "
                "(weights applied after min–max normalization "
                "of each target):"
            )
            _combo_df = pd.DataFrame(
                [
                    {
                        "Scenario": name,
                        **{
                            t: f"{w:.2f}"
                            for t, w in zip(target_cols, weights)
                        },
                    }
                    for name, weights in _combos
                ]
            )
            st.dataframe(
                _combo_df,
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "Generate multi-objective suggestions",
                type="primary",
            ):
                n_sugg = int(
                    st.session_state.get("n_suggestions", 5)
                )
                _mo: dict[
                    str, tuple[pd.DataFrame, object]
                ] = {}
                try:
                    with st.spinner(
                        "Running Bayesian Optimization "
                        "for all scenarios…"
                    ):
                        ps = build_parameter_space(
                            st.session_state.selected_features,
                            st.session_state.bounds,
                        )
                        df_train = (
                            st.session_state.processed_df
                        )
                        for _cname, _weights in _combos:
                            _df_sc = make_scalarized_target(
                                df_train,
                                target_cols,
                                _weights,
                                col_name="_combined_score",
                            )
                            _s = run_optimization(
                                ps,
                                _df_sc,
                                target_col=(
                                    "_combined_score"
                                ),
                                n_suggestions=n_sugg,
                                aux_target_cols=target_cols,
                                categorical_maps=(
                                    st.session_state.get(
                                        "categorical_maps", {}
                                    )
                                ),
                            )
                            _mo[_cname] = _s
                    st.session_state.mo_suggestions = _mo
                    st.session_state.pareto_suggestions = None
                    st.success(
                        f"Generated {n_sugg} suggestion(s) "
                        f"for {len(_combos)} scenario(s)."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Optimization failed: {exc}")

            if st.session_state.mo_suggestions:
                st.subheader("Suggested experiments")
                _mo_sugg: dict = (
                    st.session_state.mo_suggestions
                )
                _mo_tabs = st.tabs(list(_mo_sugg.keys()))
                for _mo_tab, (
                    _cname, _s_df
                ) in zip(_mo_tabs, _mo_sugg.items()):
                    with _mo_tab:
                        _show_suggestions(
                            _s_df,
                            "suggestions_"
                            f"{_cname.lower().replace(' ', '_')}"
                            ".csv",
                        )

        else:  # Pareto front (qEHVI)
            st.info(
                f"Will fit one GP per target "
                f"({', '.join(target_cols)}) and use "
                f"q-Expected Hypervolume Improvement to find "
                f"a batch of experiments on the Pareto front.",
                icon="ℹ️",
            )

            if st.button(
                "Generate Pareto-front suggestions",
                type="primary",
            ):
                n_sugg = int(
                    st.session_state.get("n_suggestions", 5)
                )
                try:
                    with st.spinner(
                        "Running Pareto-front BO (qEHVI)…"
                    ):
                        ps = build_parameter_space(
                            st.session_state.selected_features,
                            st.session_state.bounds,
                        )
                        _p_sugg = run_pareto_optimization(
                            parameters=ps,
                            training_data=(
                                st.session_state.processed_df
                            ),
                            target_cols=target_cols,
                            n_suggestions=n_sugg,
                            categorical_maps=(
                                st.session_state.get(
                                    "categorical_maps", {}
                                )
                            ),
                        )
                    st.session_state.pareto_suggestions = (
                        _p_sugg
                    )
                    st.session_state.mo_suggestions = {}
                    st.success(
                        f"Generated {n_sugg} Pareto-front "
                        f"suggestion(s)."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Pareto BO failed: {exc}")

            if st.session_state.get("pareto_suggestions") is not None:
                _p_df: pd.DataFrame = (
                    st.session_state.pareto_suggestions
                )
                st.subheader("Suggested experiments (Pareto front)")
                _show_suggestions(
                    _p_df,
                    "pareto_suggestions.csv",
                )
                # Pareto front plot with staircase + dominated region
                _pred_cols = [
                    c for c in _p_df.columns
                    if c.startswith("predicted_")
                ]
                if len(_pred_cols) >= 2:
                    # Use first decoded string column as color (e.g. Catalyst)
                    _color = next(
                        (
                            c for c in _p_df.columns
                            if pd.api.types.is_object_dtype(_p_df[c])
                        ),
                        None,
                    )
                    _fig_p = pareto_front_plot(
                        _p_df,
                        x_col=_pred_cols[0],
                        y_col=_pred_cols[1],
                        color_col=_color,
                        title="Predicted Pareto front",
                    )
                    st.plotly_chart(
                        _fig_p,
                        use_container_width=True,
                        key="pareto_front",
                    )

    else:
        # ── Single-objective ──────────────────────────────────
        if len(target_cols) > 1:
            bo_target: str = st.selectbox(
                "Optimization target",
                options=target_cols,
                key="bo_target",
            )
        else:
            bo_target = target_cols[0]

        if st.button("Generate suggestions", type="primary"):
            n_sugg = int(
                st.session_state.get("n_suggestions", 5)
            )
            try:
                with st.spinner(
                    "Running Bayesian Optimization…"
                ):
                    ps = build_parameter_space(
                        st.session_state.selected_features,
                        st.session_state.bounds,
                    )
                    df_train = st.session_state.processed_df
                    suggestions = run_optimization(
                        ps,
                        df_train,
                        target_col=bo_target,
                        n_suggestions=n_sugg,
                        categorical_maps=(
                            st.session_state.get(
                                "categorical_maps", {}
                            )
                        ),
                    )
                st.session_state.suggested_experiments = (
                    suggestions
                )
                st.success(
                    f"Generated {n_sugg} suggestion(s) "
                    f"optimizing '{bo_target}'."
                )
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                st.error(f"Optimization failed: {exc}")
                with st.expander("Show full error details"):
                    st.code(_tb.format_exc())

        if st.session_state.suggested_experiments is not None:
            st.subheader("Suggested experiments")
            _show_suggestions(
                st.session_state.suggested_experiments,
                "suggested_experiments.csv",
            )


# ---------------------------------------------------------------------------
# Step 7 — Report helpers
# ---------------------------------------------------------------------------


def _report_dataset_summary() -> dict[str, object]:
    """Collect dataset metadata for the report."""
    raw: pd.DataFrame = st.session_state.raw_df
    return {
        "file_name": st.session_state.file_name,
        "n_rows": (
            len(st.session_state.processed_df)
            if st.session_state.processed_df is not None
            else len(raw) if raw is not None else "—"
        ),
        "n_features": len(st.session_state.feature_cols),
        "target_cols": st.session_state.target_cols,
        "replicate_strategy": (
            st.session_state.replicate_strategy
        ),
    }


def _run_html_report() -> None:
    """Build the HTML report and persist the path to session state."""
    import tempfile  # noqa: PLC0415
    out_dir = Path(
        tempfile.mkdtemp(prefix="nextscreen_report_")
    )
    summary = _report_dataset_summary()
    with st.spinner("Generating HTML report…"):
        try:
            # Decode any categorical columns in the suggestions before
            # passing to the report, so the HTML shows "Pt" not 2.0.
            _sugg_raw = st.session_state.suggested_experiments
            if _sugg_raw is not None:
                _dec_maps = st.session_state.get(
                    "categorical_maps", {}
                )
                _sugg_for_report = _sugg_raw.copy()
                for _col, _lmap in _dec_maps.items():
                    if (
                        _col in _sugg_for_report.columns
                        and not pd.api.types.is_object_dtype(
                            _sugg_for_report[_col]
                        )
                    ):
                        _dec = {
                            int(v): k for k, v in _lmap.items()
                        }
                        _sugg_for_report[_col] = (
                            pd.to_numeric(
                                _sugg_for_report[_col],
                                errors="coerce",
                            )
                            .round()
                            .astype("Int64")
                            .map(_dec)
                        )
            else:
                _sugg_for_report = None

            html_path = build_html_report(
                dataset_summary=summary,
                feature_results=(
                    st.session_state.feature_results
                ),
                consensus_results=(
                    st.session_state.consensus_results
                ),
                selected_features=(
                    st.session_state.selected_features
                ),
                bounds=st.session_state.bounds,
                suggested_experiments=_sugg_for_report,
                output_dir=out_dir,
                pareto_suggestions=(
                    st.session_state.get("pareto_suggestions")
                ),
            )
            st.session_state.report_path = str(html_path)
            st.success("HTML report generated.")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")


def _run_pdf_export() -> None:
    """Convert the existing HTML report to PDF."""
    if st.session_state.report_path is None:
        st.warning(
            "Generate the HTML report first, then export to PDF."
        )
        return
    html_path = Path(st.session_state.report_path)
    with st.spinner("Converting to PDF…"):
        try:
            export_pdf(html_path)
            st.success("PDF exported successfully.")
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")


# ---------------------------------------------------------------------------
# Step 7 — Report Export
# ---------------------------------------------------------------------------

def render_step7() -> None:
    _step_header(
        "📄 Step 7 — Report Export",
        "Download a complete HTML/PDF summary of your screening "
        "session.",
    )

    if not st.session_state.selected_features:
        st.warning(
            "Complete Steps 1–5 before exporting a report."
        )
        return

    st.info(
        "The report will include: dataset summary, all variable "
        "importance and correlation plots, plain-English "
        "interpretations, chosen variables and bounds, and "
        "(when available) NEXTorch suggested experiments."
    )

    try:
        import weasyprint as _wp  # noqa: F401
        _weasyprint_available = True
    except ImportError:
        _weasyprint_available = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate HTML report", type="primary"):
            _run_html_report()
    with col2:
        if _weasyprint_available:
            if st.button("Export as PDF"):
                _run_pdf_export()
        else:
            st.info(
                "PDF export requires WeasyPrint (system libraries not "
                "available on Streamlit Cloud). Download the HTML report "
                "and open it in a browser to print/save as PDF."
            )

    if st.session_state.report_path is not None:
        rp = Path(st.session_state.report_path)
        if rp.exists():
            st.success(f"Report saved to: `{rp}`")
            html_bytes = rp.read_bytes()
            st.download_button(
                "⬇ Download HTML report",
                data=html_bytes,
                file_name=rp.name,
                mime="text/html",
            )
            pdf_path = rp.with_suffix(".pdf")
            if pdf_path.exists():
                pdf_bytes = pdf_path.read_bytes()
                st.download_button(
                    "⬇ Download PDF report",
                    data=pdf_bytes,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_RENDERERS = {
    1: render_step1,
    2: render_step2,
    3: render_step3,
    4: render_step4,
    5: render_step5,
    6: render_step6,
    7: render_step7,
}

_RENDERERS[st.session_state.step]()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Streamlit app from the ``nextscreen`` CLI command."""
    app_path = Path(__file__).resolve()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


if __name__ == "__main__":
    pass
