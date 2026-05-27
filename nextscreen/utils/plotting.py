"""Shared Plotly figure helpers used across NEXTscreen pages."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

_IMPORTANCE_COLORS: dict[str, str] = {
    "strongly important": "#2ecc71",
    "moderately important": "#f39c12",
    "weakly important or inconsistent": "#95a5a6",
}


def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
    color_col: str | None = None,
    horizontal: bool = True,
) -> go.Figure:
    """Create a styled bar chart suitable for feature importance plots.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot; must contain *x_col* and *y_col*.
    x_col : str
        Column name for the x-axis values (typically feature names).
    y_col : str
        Column name for the y-axis values (typically importance scores).
    title : str
        Plot title displayed above the chart.
    x_label : str or None, optional
        Axis label override for x. Defaults to *x_col*.
    y_label : str or None, optional
        Axis label override for y. Defaults to *y_col*.
    color_col : str or None, optional
        Column used to color-code bars (e.g., importance label). If
        ``None``, all bars share the same color.
    horizontal : bool, optional
        If ``True`` (default), bars are drawn horizontally (features on
        y-axis). Set to ``False`` for vertical bars.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly Figure ready to be passed to ``st.plotly_chart()``.
    """
    xl = x_label or x_col
    yl = y_label or y_col

    if color_col is not None and color_col in df.columns:
        colors: str | list[str] = [
            _IMPORTANCE_COLORS.get(str(v), "#3498db")
            for v in df[color_col]
        ]
    else:
        colors = "#3498db"

    if horizontal:
        # Left margin wide enough for the longest tick label.
        max_label_len = (
            max(len(str(v)) for v in df[x_col]) if not df.empty else 10
        )
        left_margin = max(120, min(max_label_len * 8, 260))
        trace = go.Bar(
            x=df[y_col],
            y=df[x_col],
            orientation="h",
            marker_color=colors,
        )
        layout = go.Layout(
            title=title,
            xaxis=dict(title=yl, automargin=True),
            yaxis=dict(autorange="reversed", automargin=True),
            height=max(350, 30 * len(df) + 120),
            margin=dict(l=left_margin, r=20, t=60, b=60),
        )
    else:
        trace = go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=colors,
        )
        layout = go.Layout(
            title=title,
            xaxis_title=xl,
            yaxis_title=yl,
            height=400,
        )

    return go.Figure(data=[trace], layout=layout)


def correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Feature\u2013Target Correlations",
) -> go.Figure:
    """Annotated heatmap of feature–target correlation coefficients.

    When passed the full output of
    :func:`~nextscreen.features.correlations.run_correlations`, only the
    ``pearson_r`` and ``spearman_r`` columns are displayed (the signed
    r-value columns, range [-1, 1]).  Categorical features assessed via
    ANOVA will have NaN for those columns and render as blank cells,
    which correctly conveys that Pearson/Spearman do not apply to them.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Either the full output of ``run_correlations`` or any DataFrame
        whose rows are features and columns are numeric metrics.
    title : str, optional
        Plot title. Default is ``'Feature–Target Correlations'``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Select only signed r-value columns when the full corr_result is passed.
    _r_cols = [c for c in ("pearson_r", "spearman_r") if c in corr_df.columns]
    display_df = corr_df[_r_cols] if _r_cols else corr_df

    z = display_df.to_numpy(dtype=float)
    x_labels = [str(c) for c in display_df.columns]
    y_labels = [str(i) for i in display_df.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1.0,
            zmax=1.0,
            text=np.where(np.isnan(z), "n/a", np.round(z, 3).astype(str)),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="r"),
        )
    )
    max_label_len = max((len(s) for s in y_labels), default=10)
    left_margin = max(120, min(max_label_len * 8, 280))
    fig.update_layout(
        title=title,
        xaxis_title="Method",
        yaxis=dict(automargin=True),
        height=max(400, 30 * len(display_df) + 100),
        margin=dict(l=left_margin, r=20, t=60, b=50),
    )
    return fig


def pca_variance_plot(
    explained_variance_ratio: np.ndarray,
    title: str = "PCA Explained Variance",
) -> go.Figure:
    """Create a scree plot of per-component and cumulative explained variance.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Per-component explained-variance ratios as returned by
        :func:`nextscreen.features.pca.run_pca`.
    title : str, optional
        Plot title. Default is ``'PCA Explained Variance'``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio) * 100
    individual = explained_variance_ratio * 100
    pc_labels = [f"PC{i + 1}" for i in range(n)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pc_labels,
            y=individual.tolist(),
            name="Per component (%)",
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pc_labels,
            y=cumulative.tolist(),
            mode="lines+markers",
            name="Cumulative (%)",
            line=dict(color="firebrick"),
            marker=dict(color="firebrick"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        yaxis=dict(range=[0, 105]),
        height=400,
    )
    return fig


def pca_loading_heatmap(
    loadings: pd.DataFrame,
    title: str = "PCA Loading Matrix",
    explained_variance_ratio: np.ndarray | None = None,
) -> go.Figure:
    """Create an annotated heatmap of PCA loading coefficients.

    Parameters
    ----------
    loadings : pd.DataFrame
        Loading matrix (features × components) as returned by
        :func:`nextscreen.features.pca.run_pca`.
    title : str, optional
        Plot title. Default is ``'PCA Loading Matrix'``.
    explained_variance_ratio : np.ndarray or None, optional
        Per-component explained-variance ratios.  When provided, column
        headers show the percentage (e.g. ``'PC1 (45.2%)'``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    z = loadings.values.astype(float)
    y_labels = loadings.index.tolist()

    if explained_variance_ratio is not None:
        evr = np.asarray(explained_variance_ratio)
        x_labels = [
            f"PC{i + 1} ({evr[i] * 100:.1f}%)"
            if i < len(evr)
            else f"PC{i + 1}"
            for i in range(len(loadings.columns))
        ]
    else:
        x_labels = loadings.columns.tolist()

    max_abs = float(np.abs(z).max()) or 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            text=np.round(z, 3),
            texttemplate="%{text:.3f}",
            showscale=True,
            colorbar=dict(
                title="Loading<br>coefficient",
                tickformat=".2f",
            ),
            hovertemplate=(
                "<b>%{y}</b> on %{x}"
                "<br>Loading = %{z:.3f}<extra></extra>"
            ),
        )
    )
    max_label_len = max((len(str(y)) for y in y_labels), default=10)
    left_margin = max(120, min(max_label_len * 8, 280))
    fig.update_layout(
        title=title,
        xaxis_title="Principal Component (% variance explained)",
        yaxis=dict(automargin=True),
        height=max(400, 35 * len(loadings) + 100),
        margin=dict(l=left_margin, r=20, t=60, b=50),
    )
    return fig


def pca_biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    explained_variance_ratio: np.ndarray | None = None,
    title: str = "PCA Biplot",
) -> go.Figure:
    """PCA biplot: sample scores in PC1/PC2 space with loading vectors.

    Parameters
    ----------
    scores : pd.DataFrame
        Projected sample coordinates (n_samples × n_components) as
        returned by :func:`nextscreen.features.pca.run_pca`.
    loadings : pd.DataFrame
        Loading matrix (n_features × n_components) from ``run_pca``.
    explained_variance_ratio : np.ndarray or None, optional
        Per-component ratios used to label axes. Default ``None``.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    x_sc = scores.iloc[:, 0].to_numpy(dtype=float)
    y_sc = (
        scores.iloc[:, 1].to_numpy(dtype=float)
        if scores.shape[1] > 1
        else np.zeros(len(scores))
    )
    x_ld = loadings.iloc[:, 0].to_numpy(dtype=float)
    y_ld = (
        loadings.iloc[:, 1].to_numpy(dtype=float)
        if loadings.shape[1] > 1
        else np.zeros(len(loadings))
    )

    # Scale loading vectors to fit within the score scatter cloud.
    sc_range = max(float(np.abs(x_sc).max()), float(np.abs(y_sc).max()), 1e-9)
    ld_range = max(float(np.abs(x_ld).max()), float(np.abs(y_ld).max()), 1e-9)
    vec_scale = sc_range * 0.80 / ld_range

    fig = go.Figure()

    # Sample scatter
    fig.add_trace(
        go.Scatter(
            x=x_sc.tolist(),
            y=y_sc.tolist(),
            mode="markers",
            marker=dict(size=8, color="steelblue", opacity=0.7),
            name="Samples",
            hovertemplate=(
                "Sample %{pointNumber}<br>"
                "PC1=%{x:.3f}<br>PC2=%{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # Loading vectors as annotations (arrows from origin)
    feature_names = loadings.index.tolist()
    for fname, xl, yl in zip(feature_names, x_ld, y_ld):
        xs = float(xl) * vec_scale
        ys = float(yl) * vec_scale
        fig.add_annotation(
            x=xs,
            y=ys,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="#e74c3c",
            text="",
        )

    # Feature labels just beyond arrowhead
    fig.add_trace(
        go.Scatter(
            x=[float(xl) * vec_scale * 1.18 for xl in x_ld],
            y=[float(yl) * vec_scale * 1.18 for yl in y_ld],
            mode="text",
            text=feature_names,
            textfont=dict(color="#c0392b", size=11),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Axis labels with variance explained
    evr = (
        np.asarray(explained_variance_ratio)
        if explained_variance_ratio is not None
        else None
    )
    pc1_label = (
        f"PC1 ({evr[0]*100:.1f}%)"
        if evr is not None and len(evr) > 0
        else "PC1"
    )
    pc2_label = (
        f"PC2 ({evr[1]*100:.1f}%)"
        if evr is not None and len(evr) > 1
        else "PC2"
    )

    fig.add_hline(y=0, line=dict(color="lightgray", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="lightgray", width=1, dash="dot"))

    fig.update_layout(
        title=title,
        xaxis_title=pc1_label,
        yaxis_title=pc2_label,
        height=520,
    )
    return fig


# def shap_beeswarm(
#     shap_values: np.ndarray,
#     X_background: pd.DataFrame,
#     title: str = "SHAP Beeswarm Plot",
# ) -> go.Figure:
#     """Create a beeswarm-style SHAP summary plot using Plotly.

#     Parameters
#     ----------
#     shap_values : np.ndarray
#         SHAP value matrix of shape (n_samples, n_features).
#     X_background : pd.DataFrame
#         Background feature values used to color-code each point by feature
#         value (low → blue, high → red).
#     title : str, optional
#         Plot title. Default is ``'SHAP Beeswarm Plot'``.

#     Returns
#     -------
#     plotly.graph_objects.Figure
#     """
#     n_samples, n_features = shap_values.shape
#     feature_names = X_background.columns.tolist()

#     mean_abs = np.abs(shap_values).mean(axis=0)
#     order = np.argsort(mean_abs)[::-1]

#     rng = np.random.default_rng(42)
#     fig = go.Figure()

#     for rank, idx in enumerate(order):
#         feat = feature_names[idx]
#         sv = shap_values[:, idx]
#         fv = X_background[feat].values.astype(float)

#         f_min, f_max = fv.min(), fv.max()
#         if f_max > f_min:
#             fv_norm = (fv - f_min) / (f_max - f_min)
#         else:
#             fv_norm = np.full_like(fv, 0.5, dtype=float)

#         jitter = rng.uniform(-0.35, 0.35, n_samples)
#         y_vals = rank + jitter

#         r_c = (fv_norm * 255).astype(int)
#         b_c = ((1.0 - fv_norm) * 255).astype(int)
#         marker_colors = [
#             f"rgba({r},{50},{b},0.7)"
#             for r, b in zip(r_c, b_c)
#         ]

#         fig.add_trace(
#             go.Scatter(
#                 x=sv.tolist(),
#                 y=y_vals.tolist(),
#                 mode="markers",
#                 marker=dict(color=marker_colors, size=4),
#                 name=feat,
#                 showlegend=False,
#             )
#         )

#     # Add a hidden scatter trace whose sole purpose is to render a colorbar
#     # representing the low (blue) → high (red) feature-value scale.
#     fig.add_trace(
#         go.Scatter(
#             x=[None],
#             y=[None],
#             mode="markers",
#             marker=dict(
#                 color=[0],
#                 colorscale=[
#                     [0.0, "rgba(0,50,255,0.7)"],
#                     [1.0, "rgba(255,50,0,0.7)"],
#                 ],
#                 cmin=0,
#                 cmax=1,
#                 showscale=True,
#                 colorbar=dict(
#                     title="Feature value",
#                     titleside="right",
#                     tickvals=[0, 1],
#                     ticktext=["Low", "High"],
#                     thickness=14,
#                     len=0.5,
#                 ),
#             ),
#             showlegend=False,
#         )
#     )

#     sorted_names = [feature_names[i] for i in order]
#     fig.update_layout(
#         title=title,
#         xaxis_title="SHAP value (impact on model output)",
#         yaxis=dict(
#             tickmode="array",
#             tickvals=list(range(n_features)),
#             ticktext=sorted_names,
#         ),
#         height=max(400, 50 * n_features + 100),
#     )
#     return fig


def shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    title: str = "SHAP Beeswarm Plot",
    *,
    max_display: Optional[int] = None,
    jitter: float = 0.35,
    point_size: int = 5,
    seed: int = 42,
) -> go.Figure:
    """
    Plotly SHAP beeswarm-style summary plot.

    Requirements for correctness:
    - shap_values is shape (n_samples, n_features)
    - X is the exact matrix used for shap_values (same rows and columns).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values with shape (n_samples, n_features).
    X : pd.DataFrame
        Feature matrix used to color points by feature value.
        Must align exactly with shap_values (rows and columns).
    title : str
        Plot title.
    max_display : int | None
        Show only top-k features by mean(|SHAP|). None -> show all.
    jitter : float
        Maximum vertical jitter amplitude.
    point_size : int
        Marker size.
    seed : int
        RNG seed for deterministic jitter.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # --- Validate inputs (hard fail on silent-wrong cases) ---
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame with feature names as columns."
        )

    sv = np.asarray(shap_values)
    if sv.ndim != 2:
        raise ValueError(
            "shap_values must be 2D (n_samples, n_features). "
            f"Got {sv.ndim}D."
        )
    n_samples, n_features = sv.shape

    if len(X) != n_samples:
        raise ValueError(
            f"Row mismatch: shap_values has {n_samples} samples "
            f"but X has {len(X)} rows. "
            "Pass the exact X (same rows, same order)."
        )
    if X.shape[1] != n_features:
        raise ValueError(
            f"Column mismatch: shap_values has {n_features} features "
            f"but X has {X.shape[1]} columns. "
            "Pass the exact X (same columns, same order)."
        )

    # Force numeric for coloring; NaN-safe.
    X_num = X.apply(pd.to_numeric, errors="coerce")

    # --- Feature ordering by importance ---
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    if max_display is not None:
        max_display = int(max_display)
        if max_display <= 0:
            raise ValueError("max_display must be a positive integer or None.")
        order = order[: min(max_display, n_features)]

    # --- Helper: density-based jitter (approx beeswarm) ---
    def _density_jitter(
        x: np.ndarray, max_j: float, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Make jitter larger in dense x-regions to reduce overplotting.
        Uses a 1D histogram estimate; stable and fast.
        """
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.isfinite(x).any():
            return rng.uniform(-max_j, max_j, size=x.size)

        x_finite = x[np.isfinite(x)]
        if x_finite.size < 5:
            return rng.uniform(-max_j, max_j, size=x.size)

        # Freedman–Diaconis-ish bins; clamp to sane range
        q25, q75 = np.percentile(x_finite, [25, 75])
        iqr = max(q75 - q25, 1e-12)
        bin_width = 2.0 * iqr / (x_finite.size ** (1.0 / 3.0))
        if not np.isfinite(bin_width) or bin_width <= 0:
            nbins = 30
        else:
            x_range = x_finite.max() - x_finite.min()
            nbins = int(np.clip(x_range / bin_width, 10, 80))

        counts, edges = np.histogram(x_finite, bins=nbins)
        # Map each x to its bin count (density proxy)
        x_clipped = np.clip(x, edges[0], edges[-1])
        bin_idx = np.clip(
            np.digitize(x_clipped, edges) - 1, 0, len(counts) - 1
        )
        dens = counts[bin_idx].astype(float)

        # Normalize density to [0,1]; higher density -> bigger jitter
        if dens.max() > dens.min():
            dens_norm = (dens - dens.min()) / (dens.max() - dens.min())
        else:
            dens_norm = np.zeros_like(dens)

        # Jitter scale in [0.15, 1.0] * max_j
        scale = (0.15 + 0.85 * dens_norm) * max_j
        return rng.uniform(-1.0, 1.0, size=x.size) * scale

    rng = np.random.default_rng(seed)
    fig = go.Figure()

    # --- Build traces ---
    for rank, j in enumerate(order):
        feat = X.columns[j]
        x_shap = sv[:, j].astype(float)

        # Feature value for color (numeric, robust)
        fv = X_num.iloc[:, j].to_numpy(dtype=float)

        # Normalize feature values to [0,1] ignoring NaNs
        fv_norm = np.full(n_samples, 0.5, dtype=float)
        mask = np.isfinite(fv)
        if mask.any():
            f_min = np.nanmin(fv[mask])
            f_max = np.nanmax(fv[mask])
            if np.isfinite(f_min) and np.isfinite(f_max) and (f_max > f_min):
                fv_norm[mask] = (fv[mask] - f_min) / (f_max - f_min)

        y = (
            np.full(n_samples, rank, dtype=float)
            + _density_jitter(x_shap, jitter, rng)
        )

        # One trace per feature; colorscale gives a shared colorbar.
        fig.add_trace(
            go.Scatter(
                x=x_shap,
                y=y,
                mode="markers",
                name=str(feat),
                showlegend=False,
                marker=dict(
                    size=point_size,
                    opacity=0.75,
                    color=fv_norm,
                    colorscale=[
                        [0.0, "rgba(0,50,255,0.8)"],
                        [1.0, "rgba(255,50,0,0.8)"],
                    ],
                    cmin=0.0,
                    cmax=1.0,
                    showscale=(rank == 0),
                    colorbar=dict(
                        title="Feature value",
                        tickvals=[0, 1],
                        ticktext=["Low", "High"],
                        thickness=14,
                        len=0.6,
                    )
                    if rank == 0
                    else None,
                ),
                hovertemplate=(
                    f"<b>{feat}</b><br>"
                    "SHAP=%{x:.5f}<br>"
                    "y=%{y:.2f}<br>"
                    "value(norm)=%{marker.color:.3f}<extra></extra>"
                ),
            )
        )

    sorted_names = [X.columns[i] for i in order]

    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (impact on model output)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(order))),
            ticktext=sorted_names,
            autorange="reversed",  # top feature at top
        ),
        height=max(420, 55 * len(order) + 120),
        margin=dict(l=170, r=40, t=70, b=50),
    )

    # Optional: center line at 0 for readability
    fig.add_vline(x=0, line_width=1)

    return fig


def _build_staircase(
    xs: np.ndarray, ys: np.ndarray, x_left: float, x_right: float,
    y_bottom: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (step_x, step_y, dom_x, dom_y) for a maximization staircase."""
    n = len(xs)
    dom_x: list[float] = [x_left, float(xs[0])]
    dom_y: list[float] = [float(ys[0]), float(ys[0])]
    step_x: list[float] = [x_left, float(xs[0])]
    step_y: list[float] = [float(ys[0]), float(ys[0])]
    for k in range(n - 1):
        dom_x += [float(xs[k]), float(xs[k + 1])]
        dom_y += [float(ys[k + 1]), float(ys[k + 1])]
        step_x += [float(xs[k]), float(xs[k + 1])]
        step_y += [float(ys[k + 1]), float(ys[k + 1])]
    step_x.append(x_right)
    step_y.append(float(ys[-1]))
    dom_x += [x_right, x_right, x_left]
    dom_y += [float(ys[-1]), y_bottom, y_bottom]
    return step_x, step_y, dom_x, dom_y


def pareto_front_plot(
    suggestions: pd.DataFrame,
    x_col: str,
    y_col: str,
    current_pareto: "pd.DataFrame | None" = None,
    x_obs_col: str | None = None,
    y_obs_col: str | None = None,
    color_col: str | None = None,
    title: str = "Objective Space",
) -> go.Figure:
    """Objective-space plot: observed Pareto front + BO candidate suggestions.

    When *current_pareto* is supplied the staircase and dominated region are
    drawn from **actual observed data**, and the BO suggestions are overlaid
    as star markers.  When omitted the function falls back to drawing the
    staircase from the suggestion predictions (legacy behaviour).

    Parameters
    ----------
    suggestions : pd.DataFrame
        BO candidate experiments; must contain *x_col* and *y_col*.
    x_col : str
        Predicted-objective column for the x-axis
        (e.g. ``'predicted_yield'``).
    y_col : str
        Predicted-objective column for the y-axis.
    current_pareto : pd.DataFrame or None, optional
        Non-dominated subset of training data in objective space
        (columns *x_obs_col* / *y_obs_col*).  When provided, drives the
        Pareto-front line.
    x_obs_col : str or None, optional
        Column in *current_pareto* for x-axis values.  Defaults to
        stripping ``'predicted_'`` from *x_col*.
    y_obs_col : str or None, optional
        Column in *current_pareto* for y-axis values.
    color_col : str or None, optional
        Column in *suggestions* used to color suggestion markers.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _x_label = x_obs_col or x_col.replace("predicted_", "")
    _y_label = y_obs_col or y_col.replace("predicted_", "")

    sugg = suggestions.copy().reset_index(drop=True)
    n_sugg = len(sugg)

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Pareto front staircase from observed data (preferred path)
    # ------------------------------------------------------------------
    use_observed = (
        current_pareto is not None
        and x_obs_col is not None
        and y_obs_col is not None
        and len(current_pareto) >= 1
        and x_obs_col in current_pareto.columns
        and y_obs_col in current_pareto.columns
    )

    if use_observed:
        pf = (
            current_pareto[[x_obs_col, y_obs_col]]
            .dropna()
            .sort_values(x_obs_col)
            .reset_index(drop=True)
        )
        pxs = pf[x_obs_col].to_numpy(dtype=float)
        pys = pf[y_obs_col].to_numpy(dtype=float)

        # Axis range covering both observed front and suggestions
        all_x = np.concatenate(
            [pxs, sugg[x_col].to_numpy(dtype=float)]
        )
        all_y = np.concatenate(
            [pys, sugg[y_col].to_numpy(dtype=float)]
        )
        x_pad = max((all_x.max() - all_x.min()) * 0.12, 1e-6)
        y_pad = max((all_y.max() - all_y.min()) * 0.12, 1e-6)
        x_left = float(all_x.min()) - x_pad
        x_right = float(all_x.max()) + x_pad
        y_bottom = float(all_y.min()) - y_pad

        if len(pf) >= 2:
            step_x, step_y, dom_x, dom_y = _build_staircase(
                pxs, pys, x_left, x_right, y_bottom
            )
            fig.add_trace(
                go.Scatter(
                    x=dom_x, y=dom_y,
                    fill="toself",
                    fillcolor="rgba(52,152,219,0.13)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Dominated region",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=step_x, y=step_y,
                    mode="lines",
                    line=dict(color="#2980b9", width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Current Pareto front",
                )
            )

        # Observed Pareto points
        fig.add_trace(
            go.Scatter(
                x=pf[x_obs_col], y=pf[y_obs_col],
                mode="markers",
                marker=dict(
                    size=9, color="#2980b9",
                    symbol="circle",
                    line=dict(color="white", width=1),
                ),
                name="Observed Pareto point",
                hovertemplate=(
                    f"{_x_label}: %{{x:.4g}}<br>"
                    f"{_y_label}: %{{y:.4g}}<extra></extra>"
                ),
            )
        )

        # BO suggestions as stars
        _palette = [
            "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12",
            "#1abc9c", "#34495e", "#e67e22",
        ]
        if color_col and color_col in sugg.columns:
            for idx, val in enumerate(sugg[color_col].unique()):
                sub = sugg[sugg[color_col] == val]
                fig.add_trace(
                    go.Scatter(
                        x=sub[x_col], y=sub[y_col],
                        mode="markers+text",
                        marker=dict(
                            size=14,
                            color=_palette[idx % len(_palette)],
                            symbol="star",
                            line=dict(color="white", width=1),
                        ),
                        text=[f"#{i + 1}" for i in sub.index],
                        textposition="top center",
                        name=f"Suggestion ({val})",
                        hovertemplate=(
                            f"#{{}}<br>"
                            f"Pred. {_x_label}: %{{x:.4g}}<br>"
                            f"Pred. {_y_label}: %{{y:.4g}}"
                            "<extra></extra>"
                        ),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=sugg[x_col], y=sugg[y_col],
                    mode="markers+text",
                    marker=dict(
                        size=14, color="#e74c3c",
                        symbol="star",
                        line=dict(color="white", width=1),
                    ),
                    text=[f"#{i + 1}" for i in range(n_sugg)],
                    textposition="top center",
                    name="Candidate suggestion",
                    hovertemplate=(
                        f"Pred. {_x_label}: %{{x:.4g}}<br>"
                        f"Pred. {_y_label}: %{{y:.4g}}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            xaxis=dict(
                title=_x_label,
                range=[x_left, x_right],
            ),
            yaxis=dict(
                title=_y_label,
                range=[y_bottom, float(all_y.max()) + y_pad],
            ),
        )

    # ------------------------------------------------------------------
    # Fallback: staircase from suggestion predictions (legacy)
    # ------------------------------------------------------------------
    else:
        df = sugg.sort_values(x_col).reset_index(drop=True)
        xs = df[x_col].to_numpy(dtype=float)
        ys = df[y_col].to_numpy(dtype=float)
        n = len(xs)

        x_span = float(xs.max() - xs.min()) if n > 1 else 1.0
        y_span = float(ys.max() - ys.min()) if n > 1 else 1.0
        x_pad = max(x_span * 0.12, 1e-6)
        y_pad = max(y_span * 0.12, 1e-6)
        x_left = float(xs.min()) - x_pad
        x_right = float(xs.max()) + x_pad
        y_bottom = float(ys.min()) - y_pad

        if n >= 2:
            step_x, step_y, dom_x, dom_y = _build_staircase(
                xs, ys, x_left, x_right, y_bottom
            )
            fig.add_trace(
                go.Scatter(
                    x=dom_x, y=dom_y,
                    fill="toself",
                    fillcolor="rgba(52,152,219,0.13)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Dominated region",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=step_x, y=step_y,
                    mode="lines",
                    line=dict(color="#2980b9", width=2),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Pareto front",
                )
            )

        _palette = [
            "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12",
            "#1abc9c", "#34495e", "#e67e22",
        ]
        if color_col and color_col in df.columns:
            for idx, val in enumerate(df[color_col].unique()):
                sub = df[df[color_col] == val]
                fig.add_trace(
                    go.Scatter(
                        x=sub[x_col], y=sub[y_col],
                        mode="markers+text",
                        marker=dict(
                            size=12,
                            color=_palette[idx % len(_palette)],
                            line=dict(color="white", width=1.5),
                        ),
                        text=[f"#{i + 1}" for i in sub.index],
                        textposition="top center",
                        name=str(val),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col], y=df[y_col],
                    mode="markers+text",
                    marker=dict(
                        size=12, color="#e74c3c",
                        line=dict(color="white", width=1.5),
                    ),
                    text=[f"#{i + 1}" for i in range(n)],
                    textposition="top center",
                    name="Suggestion",
                )
            )

        fig.update_layout(
            xaxis_title=x_col.replace("predicted_", "Predicted "),
            yaxis_title=y_col.replace("predicted_", "Predicted "),
        )

    fig.update_layout(
        title=title,
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.18,
            xanchor="center",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(b=90),
        height=520,
    )
    return fig
