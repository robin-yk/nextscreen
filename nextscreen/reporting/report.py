"""HTML and PDF report generation from NEXTscreen session results."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd

from nextscreen.interpretation.narrator import (
    interpret_consensus,
    interpret_correlations,
    interpret_lasso,
    interpret_pca,
    interpret_random_forest,
    interpret_shap,
)
from nextscreen.utils.plotting import (
    bar_chart,
    correlation_heatmap,
    pca_loading_heatmap,
    pca_variance_plot,
    shap_beeswarm,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS stylesheet
# ---------------------------------------------------------------------------

_CSS = """\
body {
    font-family: Arial, Helvetica, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 28px;
    color: #2c3e50;
    line-height: 1.6;
}
h1 {
    color: #1a252f;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}
h2 {
    color: #2c3e50;
    border-left: 5px solid #3498db;
    padding-left: 12px;
    margin-top: 44px;
}
h3 { color: #34495e; margin-top: 32px; }
h4 { color: #7f8c8d; margin-top: 20px; }
.meta { color: #95a5a6; font-size: 0.88em; margin-bottom: 24px; }
.section { margin-bottom: 44px; }
.interpretation {
    background: #eaf4fb;
    border-left: 4px solid #3498db;
    padding: 12px 18px;
    margin: 14px 0;
    border-radius: 4px;
    font-style: italic;
    color: #2c3e50;
}
img.plot {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px 0;
    border: 1px solid #ecf0f1;
    border-radius: 4px;
}
table.data-table {
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 0.90em;
}
table.data-table th {
    background: #3498db;
    color: white;
    padding: 8px 12px;
    text-align: left;
}
table.data-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #ecf0f1;
}
table.data-table tr:nth-child(even) td {
    background: #f4f8fb;
}
.label-strong  { color: #27ae60; font-weight: bold; }
.label-moderate { color: #e67e22; font-weight: bold; }
.label-weak    { color: #95a5a6; }
.no-data {
    color: #95a5a6;
    font-style: italic;
    padding: 8px 0;
}
.plot-container {
    margin: 16px 0;
    border: 1px solid #ecf0f1;
    border-radius: 4px;
    overflow: hidden;
}
"""

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>NEXTscreen Feature Selection Report</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
  <style>
{css}
  </style>
</head>
<body>
{body}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fig_to_html_div(fig) -> str:
    """Render a Plotly figure as an inline HTML div (no external deps)."""
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _try_plot(fig_callable, label: str) -> str:
    """Attempt to render a plot; return a fallback notice on failure."""
    try:
        fig = fig_callable()
        div = _fig_to_html_div(fig)
        return f'<div class="plot-container">\n{div}\n</div>\n'
    except Exception as exc:
        logger.warning("Plot '%s' failed: %s", label, exc)
        return (
            f'<p class="no-data">'
            f"[Plot unavailable: {label}]"
            f"</p>"
        )


def _df_html(df: pd.DataFrame, *, index: bool = False) -> str:
    """Render a DataFrame as a styled HTML table."""
    return df.to_html(
        index=index,
        border=0,
        classes="data-table",
        escape=True,
    )


def _interp_html(text: str) -> str:
    """Wrap an interpretation string in a styled ``<div>``."""
    return f'<div class="interpretation">{text}</div>\n'


def _section(heading: str, content: str) -> str:
    """Wrap *content* in a numbered ``<section>`` div."""
    return (
        f'<section class="section">'
        f"<h2>{heading}</h2>"
        f"{content}"
        f"</section>\n"
    )


# ---------------------------------------------------------------------------
# Per-method HTML builders
# ---------------------------------------------------------------------------


def _lasso_html(result: pd.DataFrame, target: str) -> str:
    """Build the LASSO subsection HTML."""
    parts: list[str] = [f"<h4>LASSO — {target}</h4>\n"]
    parts.append(
        _try_plot(
            lambda: bar_chart(
                result,
                "feature",
                "coefficient",
                f"LASSO Coefficients — {target}",
                x_label="Feature",
                y_label="Regression coefficient",
            ),
            "LASSO coefficients",
        )
    )
    parts.append(
        _df_html(result[["feature", "coefficient", "rank"]])
    )
    try:
        parts.append(_interp_html(interpret_lasso(result, target)))
    except Exception as exc:
        logger.warning("LASSO interpretation failed: %s", exc)
    return "\n".join(parts)


def _rf_html(result: pd.DataFrame, target: str) -> str:
    """Build the Random Forest subsection HTML."""
    parts: list[str] = [f"<h4>Random Forest — {target}</h4>\n"]
    parts.append(
        _try_plot(
            lambda: bar_chart(
                result,
                "feature",
                "importance",
                f"RF Feature Importances — {target}",
                x_label="Feature",
                y_label="Importance",
            ),
            "RF importances",
        )
    )
    parts.append(
        _df_html(result[["feature", "importance", "rank"]])
    )
    try:
        parts.append(
            _interp_html(interpret_random_forest(result, target))
        )
    except Exception as exc:
        logger.warning("RF interpretation failed: %s", exc)
    return "\n".join(parts)


def _shap_html(result: dict[str, object], target: str) -> str:
    """Build the SHAP subsection HTML."""
    parts: list[str] = [f"<h4>SHAP — {target}</h4>\n"]
    fi: pd.DataFrame = result["feature_importance"]  # type: ignore[assignment]
    parts.append(
        _try_plot(
            lambda: bar_chart(
                fi,
                "feature",
                "mean_abs_shap",
                f"SHAP Feature Importances — {target}",
                x_label="Feature",
                y_label="Mean |SHAP|",
            ),
            "SHAP importances",
        )
    )
    sv = result.get("shap_values")
    X_bg = result.get("X_background")
    if sv is not None and X_bg is not None:
        parts.append(
            _try_plot(
                lambda: shap_beeswarm(
                    sv,  # type: ignore[arg-type]
                    X_bg,  # type: ignore[arg-type]
                    title=f"SHAP Beeswarm — {target}",
                ),
                "SHAP beeswarm",
            )
        )
    parts.append(_df_html(fi[["feature", "mean_abs_shap", "rank"]]))
    try:
        parts.append(_interp_html(interpret_shap(result, target)))
    except Exception as exc:
        logger.warning("SHAP interpretation failed: %s", exc)
    return "\n".join(parts)


def _pca_html(result: dict[str, object], target: str) -> str:
    """Build the PCA subsection HTML."""
    parts: list[str] = [f"<h4>PCA — {target}</h4>\n"]
    evr = result.get("explained_variance_ratio")
    if evr is not None:
        parts.append(
            _try_plot(
                lambda: pca_variance_plot(
                    evr,  # type: ignore[arg-type]
                    title="PCA Explained Variance",
                ),
                "PCA variance",
            )
        )
    loadings = result.get("loadings")
    if loadings is not None:
        parts.append(
            _try_plot(
                lambda: pca_loading_heatmap(
                    loadings,  # type: ignore[arg-type]
                    title="PCA Component Loadings",
                ),
                "PCA loadings",
            )
        )
    fr: pd.DataFrame = result["feature_rank"]  # type: ignore[assignment]
    parts.append(_df_html(fr[["feature", "max_loading", "rank"]]))
    try:
        parts.append(_interp_html(interpret_pca(result)))
    except Exception as exc:
        logger.warning("PCA interpretation failed: %s", exc)
    return "\n".join(parts)


def _corr_html(result: pd.DataFrame, target: str) -> str:
    """Build the Correlations subsection HTML."""
    parts: list[str] = [f"<h4>Correlations — {target}</h4>\n"]
    hm_data: dict[str, object] = {}
    if "pearson_r" in result.columns:
        hm_data["Pearson r"] = result["pearson_r"]
    if "spearman_r" in result.columns:
        hm_data["Spearman ρ"] = result["spearman_r"]

    if hm_data:
        hm_df = pd.DataFrame(hm_data)
        parts.append(
            _try_plot(
                lambda: correlation_heatmap(
                    hm_df,
                    title=(
                        f"Feature\u2013Target Correlations"
                        f" — {target}"
                    ),
                ),
                "Correlations heatmap",
            )
        )

    display_cols = [
        c
        for c in result.columns
        if c
        in (
            "pearson_r",
            "pearson_p",
            "pearson_significant",
            "spearman_r",
            "spearman_p",
            "spearman_significant",
            "rank",
        )
    ]
    parts.append(_df_html(result[display_cols].reset_index()))
    try:
        parts.append(
            _interp_html(interpret_correlations(result, target))
        )
    except Exception as exc:
        logger.warning("Correlations interpretation failed: %s", exc)
    return "\n".join(parts)


def _consensus_html(
    consensus_df: pd.DataFrame,
    target: str,
    n_methods: int,
) -> str:
    """Build the Consensus Ranking subsection HTML."""
    parts: list[str] = [f"<h4>Consensus Ranking — {target}</h4>\n"]
    parts.append(
        _try_plot(
            lambda: bar_chart(
                consensus_df,
                "feature",
                "n_methods_top_k",
                f"Consensus — {target}",
                x_label="Feature",
                y_label="# Methods ranking in top-K",
                color_col="importance_label",
            ),
            "Consensus ranking",
        )
    )
    disp_cols = [
        c
        for c in [
            "feature",
            "consensus_rank",
            "n_methods_top_k",
            "importance_label",
            "avg_normalized_rank",
        ]
        if c in consensus_df.columns
    ]
    parts.append(_df_html(consensus_df[disp_cols]))
    try:
        parts.append(
            _interp_html(
                interpret_consensus(
                    consensus_df, target, n_methods
                )
            )
        )
    except Exception as exc:
        logger.warning("Consensus interpretation failed: %s", exc)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-target HTML builder
# ---------------------------------------------------------------------------

_METHOD_BUILDERS = {
    "lasso": _lasso_html,
    "random_forest": _rf_html,
    "shap": _shap_html,
    "pca": _pca_html,
    "correlations": _corr_html,
}


def _target_html(
    target: str,
    results: dict[str, object],
    consensus_df: pd.DataFrame | None,
) -> str:
    """Render all method subsections for one target variable."""
    parts: list[str] = [f"<h3>Target: {target}</h3>\n"]
    for method, builder in _METHOD_BUILDERS.items():
        result = results.get(method)
        if result is None:
            continue
        try:
            parts.append(builder(result, target))  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning(
                "Section %s/%s failed: %s", target, method, exc
            )
    if consensus_df is not None:
        n_methods = len(
            [k for k in results if k in _METHOD_BUILDERS]
        )
        try:
            parts.append(
                _consensus_html(consensus_df, target, n_methods)
            )
        except Exception as exc:
            logger.warning(
                "Consensus section %s failed: %s", target, exc
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_html_report(
    dataset_summary: dict[str, object],
    feature_results: dict[str, dict[str, object]],
    consensus_results: dict[str, pd.DataFrame],
    selected_features: list[str],
    bounds: dict[str, dict[str, float]],
    suggested_experiments: pd.DataFrame | None,
    output_dir: Path,
    pareto_suggestions: pd.DataFrame | None = None,
) -> Path:
    """Render a self-contained HTML report and write it to *output_dir*.

    All Plotly figures are embedded as base64-encoded PNGs (no external
    dependencies) so the resulting file is fully portable.

    Parameters
    ----------
    dataset_summary : dict
        Summary metadata with keys ``'file_name'``, ``'n_rows'``,
        ``'n_features'``, ``'target_cols'``, ``'replicate_strategy'``.
    feature_results : dict
        Nested mapping ``{target: {method: result}}`` as stored in
        ``st.session_state.feature_results``.
    consensus_results : dict
        Mapping ``{target: consensus_df}`` as stored in
        ``st.session_state.consensus_results``.
    selected_features : list of str
        Features chosen by the user for optimization.
    bounds : dict
        ``{feature: {'lower': float, 'upper': float, 'type': str}}``.
    suggested_experiments : pd.DataFrame or None
        Scalarized single-objective BO output; ``None`` when skipped.
    output_dir : Path
        Directory where the HTML file will be written (created if absent).
    pareto_suggestions : pd.DataFrame or None, optional
        Pareto-front (qEHVI) BO output; ``None`` when skipped.

    Returns
    -------
    Path
        Absolute path of the generated HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sections: list[str] = []

    # -- Header ---------------------------------------------------------------
    header = (
        f"<h1>NEXTscreen \u2014 Feature Selection Report</h1>\n"
        f'<p class="meta">Generated: {ts}</p>\n'
    )

    # -- Section 1: Dataset Summary -------------------------------------------
    summary_rows = [
        ("File", str(dataset_summary.get("file_name", "—"))),
        ("Rows", str(dataset_summary.get("n_rows", "—"))),
        ("Features", str(dataset_summary.get("n_features", "—"))),
        (
            "Target(s)",
            ", ".join(
                str(t)
                for t in dataset_summary.get(  # type: ignore[union-attr]
                    "target_cols", []
                )
            )
            or "—",
        ),
        (
            "Replicate strategy",
            str(
                dataset_summary.get("replicate_strategy", "—")
            ),
        ),
    ]
    summary_df = pd.DataFrame(
        summary_rows, columns=["Parameter", "Value"]
    )
    sections.append(
        _section(
            "1. Dataset Summary",
            _df_html(summary_df),
        )
    )

    # -- Section 2: Feature Selection Results ---------------------------------
    results_html: list[str] = []
    for target, results in feature_results.items():
        con_df = consensus_results.get(target)
        results_html.append(
            _target_html(target, results, con_df)
        )
    sections.append(
        _section(
            "2. Feature Selection Results",
            "\n".join(results_html),
        )
    )

    # -- Section 3: Selected Features and Bounds ------------------------------
    if selected_features:
        bounds_rows = []
        for feat in selected_features:
            b = bounds.get(feat, {})
            bounds_rows.append(
                {
                    "Feature": feat,
                    "Lower": b.get("lower", "—"),
                    "Upper": b.get("upper", "—"),
                    "Type": b.get("type", "continuous"),
                }
            )
        bounds_df = pd.DataFrame(bounds_rows)
        sections.append(
            _section(
                "3. Selected Features and Bounds",
                _df_html(bounds_df),
            )
        )

    # -- Section 4: Suggested Experiments (optional) -------------------------
    if suggested_experiments is not None:
        sections.append(
            _section(
                "4. Suggested Experiments (Weighted Scalarization BO)",
                _df_html(
                    suggested_experiments.apply(
                        lambda c: c.round(6)
                        if pd.api.types.is_numeric_dtype(c)
                        else c
                    )
                ),
            )
        )

    # -- Section 4b: Pareto-front suggestions (optional) ---------------------
    if pareto_suggestions is not None:
        sections.append(
            _section(
                "4b. Suggested Experiments (Pareto-front qEHVI)",
                _df_html(
                    pareto_suggestions.apply(
                        lambda c: c.round(6)
                        if pd.api.types.is_numeric_dtype(c)
                        else c
                    )
                ),
            )
        )

    body = header + "\n".join(sections)
    html = _HTML_TEMPLATE.format(css=_CSS, body=body)

    html_path = output_dir / "nextscreen_report.html"
    html_path.write_text(html, encoding="utf-8")

    logger.info("HTML report written to %s.", html_path)
    return html_path.resolve()


def export_pdf(
    html_path: Path, pdf_path: Path | None = None
) -> Path:
    """Convert an HTML report to PDF using WeasyPrint.

    Parameters
    ----------
    html_path : Path
        Path to the HTML report produced by :func:`build_html_report`.
    pdf_path : Path or None, optional
        Destination for the PDF. Defaults to the same directory and stem
        as *html_path* with a ``.pdf`` extension.

    Returns
    -------
    Path
        Absolute path of the generated PDF file.

    Raises
    ------
    RuntimeError
        If WeasyPrint cannot load its required system libraries (e.g.,
        ``libpango`` on macOS). Includes installation instructions.
    """
    try:
        from weasyprint import HTML as WeasyHTML  # noqa: PLC0415
    except OSError as exc:
        raise RuntimeError(
            "WeasyPrint could not load required system libraries. "
            "On macOS, install pango via Homebrew: "
            "`brew install pango`. "
            "Full instructions: https://doc.courtbouillon.org/"
            "weasyprint/stable/first_steps.html#installation"
        ) from exc

    html_path = Path(html_path).resolve()
    if pdf_path is None:
        pdf_path = html_path.with_suffix(".pdf")
    else:
        pdf_path = Path(pdf_path).resolve()

    WeasyHTML(filename=str(html_path)).write_pdf(str(pdf_path))
    logger.info("PDF report written to %s.", pdf_path)
    return pdf_path
