"""QC report rendering helpers shared by CLI orchestrations."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px

__all__ = ["generate_qc_report"]

SEGMENT_ENTITY_PREFIX = {
    "absolute": "absolute_power",
    "relative": "relative_power",
    "ratio": "power_ratio",
    "perm_entropy": "permutation_entropy",
    "spectral_entropy": "spectral_entropy",
}


def _render_table(df: pd.DataFrame, empty_message: str = "No data available.") -> str:
    if df.empty:
        return f"<p>{empty_message}</p>"
    return df.to_html(index=False, classes="table")


def _build_histogram_html(
    data: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    color: str | None = None,
    nbins: int = 30,
) -> str:
    fig = px.histogram(
        data,
        x=value_column,
        color=color,
        nbins=nbins,
        barmode="overlay",
        opacity=0.85,
        template="simple_white",
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), legend_title_text="")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def _compose_segment_entity_id(metric: str, band: str | float | None) -> Optional[str]:
    """Mirror CLI entity naming to locate per-feature segment tables."""
    prefix = SEGMENT_ENTITY_PREFIX.get(metric)
    if prefix is None:
        return None
    if band is None or (isinstance(band, float) and pd.isna(band)):
        return prefix
    return f"{prefix}[{band}]"


def _collect_segment_tables(segment_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Organize segment values by entity → subject for quick lookups."""
    if segment_df is None or segment_df.empty:
        return {}
    segment_columns = [column for column in segment_df.columns if column.startswith("segment_")]
    if not segment_columns:
        return {}
    lookup: Dict[str, Dict[str, pd.DataFrame]] = {}
    grouped = segment_df.groupby(["entity", "subject_id"])
    for (entity, subject_id), group in grouped:
        table = (
            group.sort_values("channel")
            .set_index("channel", drop=True)[segment_columns]
            .copy()
        )
        lookup.setdefault(entity, {})[subject_id] = table
    return lookup


def _build_segment_heatmap_html(table: pd.DataFrame, title: str) -> str:
    """Render a heatmap of channel × segment values."""
    fig = px.imshow(
        table,
        aspect="auto",
        labels={"x": "Segment", "y": "Channel", "color": "Value"},
        title=title,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=30))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def _build_segment_widget(
    feature_id: str,
    title: str,
    subject_tables: Dict[str, pd.DataFrame],
) -> str:
    """Build subject selector + per-subject heatmaps for segmented runs."""
    select_id = f"segment-select-{feature_id}"
    options_html = "\n".join(
        f'<option value="{subject}">{subject}</option>' for subject in sorted(subject_tables.keys())
    )
    heatmap_sections = []
    for subject in sorted(subject_tables.keys()):
        heatmap_html = _build_segment_heatmap_html(
            subject_tables[subject],
            f"{title} - Segments ({subject})",
        )
        heatmap_sections.append(
            f'<div class="segment-heatmap" data-subject="{subject}">{heatmap_html}</div>'
        )
    heatmap_html = "\n".join(heatmap_sections)
    return f"""
        <article class="card segment-widget" data-feature="{feature_id}">
            <div class="segment-controls">
                <label for="{select_id}"><strong>Segment heatmap subject:</strong></label>
                <select id="{select_id}" class="segment-select">
                    {options_html}
                </select>
            </div>
            <div class="segment-heatmaps">
                {heatmap_html}
            </div>
        </article>
    """


def _compute_zscores(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def _build_summary_view(metadata_rows: List[Dict[str, float]], tidy_df: pd.DataFrame) -> Dict[str, str]:
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_html = _render_table(metadata_df, "No EEG header metadata recorded.")

    feature_coverage = pd.DataFrame()
    if not tidy_df.empty:
        feature_coverage = (
            tidy_df.groupby(["metric", "band"])
            .agg(
                subjects=("subject_id", "nunique"),
                channels=("channel", "nunique"),
                observations=("power", "size"),
            )
            .reset_index()
            .sort_values(["metric", "band"])
        )
    coverage_html = _render_table(feature_coverage, "No feature measurements available yet.")

    summary_content = f"""
        <article class="card">
            <h2>EEG Recording Parameters</h2>
            <p>Summary of header-level parameters extracted from each input file.</p>
            {metadata_html}
        </article>
        <article class="card">
            <h2>Feature Coverage</h2>
            <p>Number of subjects, channels, and total observations contributing to each feature.</p>
            {coverage_html}
        </article>
    """
    return {"id": "summary", "label": "EEG Parameters", "content": summary_content}


def _build_feature_view(
    feature_id: str,
    metric: str,
    band: str | float | None,
    subset: pd.DataFrame,
    *,
    subject_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, str]:
    band_label = "" if band is None or (isinstance(band, float) and pd.isna(band)) else str(band)
    if band_label:
        title_prefix = band_label.title()
        title = f"{title_prefix} - {metric.title()}"
    else:
        title_prefix = metric.title()
        title = metric.title()
    channel_values = subset[["subject_id", "channel", "power"]].copy()
    channel_values["z_score"] = _compute_zscores(channel_values["power"])
    channel_values["status"] = channel_values["z_score"].abs() > 3
    channel_values["status"] = channel_values["status"].map({True: "Outlier", False: "Within Range"})
    channel_values["power"] = channel_values["power"].round(6)
    channel_values["z_score"] = channel_values["z_score"].round(3)

    subject_means = (
        subset.groupby("subject_id", as_index=False)["power"]
        .mean()
        .rename(columns={"power": "mean_power"})
    )
    subject_means["z_score"] = _compute_zscores(subject_means["mean_power"])
    subject_means["status"] = subject_means["z_score"].abs() > 3
    subject_means["status"] = subject_means["status"].map({True: "Outlier", False: "Within Range"})
    subject_means["mean_power"] = subject_means["mean_power"].round(6)
    subject_means["z_score"] = subject_means["z_score"].round(3)

    summary_stats = (
        subset["power"]
        .describe()
        .rename_axis("statistic")
        .reset_index(name="value")
    )
    summary_stats["value"] = summary_stats["value"].round(6)

    channel_hist = _build_histogram_html(
        channel_values,
        value_column="power",
        title=f"{title} - Channel Distribution",
        color="status",
    )
    subject_hist = _build_histogram_html(
        subject_means,
        value_column="mean_power",
        title=f"{title} - Subject Mean Distribution",
        color="status",
    )

    channel_table_html = _render_table(
        channel_values.rename(columns={"power": "value", "status": "outlier_status"}),
        "No per-channel values were computed.",
    )
    subject_table_html = _render_table(
        subject_means.rename(columns={"mean_power": "value", "status": "outlier_status"}),
        "No subject-level aggregates were computed.",
    )
    summary_table_html = _render_table(summary_stats, "No descriptive statistics available.")
    segment_widget_html = ""
    if subject_tables:
        segment_widget_html = _build_segment_widget(feature_id, title_prefix or metric.title(), subject_tables)

    content = f"""
        <article class="card">
            <h2>{title}</h2>
            <p>Channel-level and subject-level quality control for this feature.</p>
        </article>
        <section class="grid">
            <article class="card">
                <h3>Channel-Level Distribution</h3>
                {channel_hist}
                <h4>Channel Measurements</h4>
                {channel_table_html}
            </article>
            <article class="card">
                <h3>Subject Mean Distribution</h3>
                {subject_hist}
                <h4>Subject Mean Values</h4>
                {subject_table_html}
            </article>
        </section>
        {segment_widget_html}
        <article class="card">
            <h3>Summary Statistics</h3>
            {summary_table_html}
        </article>
    """
    return {"id": feature_id, "label": title, "content": content}


def generate_qc_report(
    metadata_rows: List[Dict[str, float]],
    tidy_df: pd.DataFrame,
    report_cfg: Dict[str, str],
    *,
    segment_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render the HTML QC report by composing summary + per-feature views."""
    views: List[Dict[str, str]] = []
    views.append(_build_summary_view(metadata_rows, tidy_df))

    segment_tables = _collect_segment_tables(segment_df)

    if tidy_df.empty:
        placeholder_content = """
            <article class="card">
                <h2>No Feature Data</h2>
                <p>The pipeline did not compute any features. Add EEG recordings and rerun the pipeline to populate this report.</p>
            </article>
        """
        views.append({"id": "no-data", "label": "Features", "content": placeholder_content})
    else:
        combos = (
            tidy_df[["metric", "band"]]
            .drop_duplicates()
            .assign(_band_sort=lambda frame: frame["band"].fillna(""))
            .sort_values(["metric", "_band_sort"])
            .drop(columns="_band_sort")
        )
        for _, row in combos.iterrows():
            metric = row["metric"]
            band = row["band"]
            if pd.isna(metric):
                continue
            if pd.isna(band):
                subset = tidy_df[(tidy_df["metric"] == metric) & (tidy_df["band"].isna())]
            else:
                subset = tidy_df[(tidy_df["metric"] == metric) & (tidy_df["band"] == band)]
            if subset.empty:
                continue
            band_slug = "overall" if pd.isna(band) else str(band)
            feature_id = f"{metric}-{band_slug}".replace(" ", "_")
            entity_id = _compose_segment_entity_id(metric, None if pd.isna(band) else band)
            subject_tables = segment_tables.get(entity_id, None) if entity_id else None
            views.append(
                _build_feature_view(
                    feature_id,
                    metric,
                    None if pd.isna(band) else band,
                    subset,
                    subject_tables=subject_tables,
                )
            )

    title = report_cfg.get("title", "qEEG QC Report")
    author = report_cfg.get("author", "unknown")
    options_html = "\n".join(
        f'<option value="{view["id"]}">{view["label"]}</option>' for view in views
    )
    view_sections_html = "\n".join(
        f'<div class="view{" active" if idx == 0 else ""}" data-view="{view["id"]}">{view["content"]}</div>'
        for idx, view in enumerate(views)
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        :root {{
            font-family: "Segoe UI", Arial, sans-serif;
            background-color: #f5f5f5;
            color: #111;
        }}
        body {{ margin: 0; padding: 1.5rem; }}
        header {{ background: #0f6cbd; color: #fff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 10px 25px rgba(15,108,189,0.25); margin-bottom: 1.5rem; }}
        header h1 {{ margin: 0 0 0.25rem; }}
        header p {{ margin: 0.25rem 0; opacity: 0.9; }}
        .controls {{ margin-bottom: 1rem; }}
        select {{
            padding: 0.6rem 1rem;
            border-radius: 6px;
            border: 1px solid #d0d7de;
            font-size: 1rem;
            min-width: 18rem;
        }}
        .view {{ display: none; }}
        .view.active {{ display: block; }}
        .card {{
            background: #fff;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 32px rgba(15,16,18,0.08);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}
        .segment-widget {{
            margin-bottom: 1.25rem;
        }}
        .segment-controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: center;
        }}
        .segment-heatmaps {{
            margin-top: 0.75rem;
        }}
        .segment-heatmap {{
            display: none;
        }}
        .segment-heatmap.active {{
            display: block;
        }}
        .table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.92rem;
        }}
        .table th, .table td {{
            padding: 0.4rem 0.6rem;
            border: 1px solid #e3e8ef;
            text-align: left;
        }}
        .table tr:nth-child(even) {{ background: #f9fafb; }}
        h2, h3, h4 {{ margin-top: 0; }}
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            var select = document.getElementById("view-select");
            var views = document.querySelectorAll(".view");
            function updateView() {{
                views.forEach(function(view) {{
                    if (view.dataset.view === select.value) {{
                        view.classList.add("active");
                    }} else {{
                        view.classList.remove("active");
                    }}
                }});
            }}
            select.addEventListener("change", updateView);
            updateView();
            function initSegmentWidgets() {{
                document.querySelectorAll(".segment-widget").forEach(function(widget) {{
                    var segmentSelect = widget.querySelector(".segment-select");
                    var panels = widget.querySelectorAll(".segment-heatmap");
                    if (!segmentSelect || panels.length === 0) {{
                        return;
                    }}
                    function updateSegments() {{
                        panels.forEach(function(panel) {{
                            if (panel.dataset.subject === segmentSelect.value) {{
                                panel.classList.add("active");
                            }} else {{
                                panel.classList.remove("active");
                            }}
                        }});
                    }}
                    segmentSelect.addEventListener("change", updateSegments);
                    updateSegments();
                }});
            }}
            initSegmentWidgets();
        }});
    </script>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <p>Author: {author}</p>
        <p>Total recordings: {len(metadata_rows)}</p>
    </header>
    <div class="controls">
        <label for="view-select"><strong>Choose QC view:</strong></label>
        <select id="view-select">
            {options_html}
        </select>
    </div>
    <section id="view-container">
        {view_sections_html}
    </section>
</body>
</html>"""
