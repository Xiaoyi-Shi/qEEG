"""QC report rendering helpers shared by CLI orchestrations."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px

__all__ = ["generate_qc_report"]


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


def _build_feature_view(feature_id: str, metric: str, band: str, subset: pd.DataFrame) -> Dict[str, str]:
    title = f"{band.title()} - {metric.title()}"
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
) -> str:
    """Render the HTML QC report by composing summary + per-feature views."""
    views: List[Dict[str, str]] = []
    views.append(_build_summary_view(metadata_rows, tidy_df))

    if tidy_df.empty:
        placeholder_content = """
            <article class="card">
                <h2>No Feature Data</h2>
                <p>The pipeline did not compute any features. Add EEG recordings and rerun the pipeline to populate this report.</p>
            </article>
        """
        views.append({"id": "no-data", "label": "Features", "content": placeholder_content})
    else:
        for metric in sorted(tidy_df["metric"].unique()):
            for band in sorted(tidy_df["band"].dropna().unique()):
                subset = tidy_df[(tidy_df["metric"] == metric) & (tidy_df["band"] == band)]
                if subset.empty:
                    continue
                feature_id = f"{metric}-{band}".replace(" ", "_")
                views.append(_build_feature_view(feature_id, metric, band, subset))

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
