"""CLI entry point for qEEG processing tasks."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import plotly.express as px

from utils.basefun import (
    compute_absolute_power,
    compute_relative_power,
    load_raw_file,
    summarise_recording,
    tidy_power_table,
    WelchParams,
)
from utils.entropy import PermEntropyParams, compute_permutation_entropy

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
SUPPORTED_FEATURES = {"absolute_power", "relative_power", "permutation_entropy"}
FEATURE_COLUMNS = ["subject_id", "channel", "band", "metric", "power"]
EMPTY_FEATURE_FRAME = pd.DataFrame(columns=FEATURE_COLUMNS)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Configure CLI flags for running the qEEG pipeline."""
    parser = argparse.ArgumentParser(description="Run the qEEG processing pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cal_qEEG_all.json"),
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data directory defined in the config file.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Override the output directory defined in the config file.",
    )
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        choices=sorted(SUPPORTED_FEATURES),
        help="Restrict processing to the named feature(s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip feature computation after discovery/logging.",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path) -> Dict:
    """Load pipeline configuration from disk."""
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_path(path_str: str, base: Path) -> Path:
    """Resolve relative paths declared inside config JSON files."""
    path = Path(path_str)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def discover_recordings(data_dir: Path) -> List[Path]:
    """Return a list of supported EEG recording files."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.fif")) + sorted(data_dir.glob("*.edf"))
    return files


def ensure_output_tree(output_dir: Path) -> Dict[str, Path]:
    """Create the timestamped output tree for the current pipeline run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": output_dir,
        "results_csv": output_dir / "qEEG_result.csv",
        "qc_html": output_dir / "QC.html",
        "log_file": log_dir / "pipeline.log",
    }


def configure_logging(log_file: Path, level: str) -> None:
    """Send structured logging to both console and the run-specific log file."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


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


def _select_feature_flags(requested: List[str] | None) -> Dict[str, bool]:
    """Normalize CLI feature selections into boolean flags."""
    if not requested:
        return {name: True for name in SUPPORTED_FEATURES}
    flags = {name: name in requested for name in SUPPORTED_FEATURES}
    if flags["relative_power"]:
        flags["absolute_power"] = True
    return flags


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    base_dir = config_path.parent

    paths_cfg = config.get("paths") or {}
    data_dir = args.data_dir or resolve_path(paths_cfg.get("data_dir", "data/EEG_DATA"), base_dir)
    output_root = args.result_dir or resolve_path(paths_cfg.get("output_dir", "result"), base_dir)

    current_time = datetime.now()
    output_dir = output_root / current_time.strftime("%Y-%m-%d-%H-%M")
    paths = ensure_output_tree(output_dir)
    configure_logging(paths["log_file"], args.log_level)

    logging.info("Starting qEEG pipeline (config=%s)", config_path)
    logging.info("Data directory: %s", data_dir)
    logging.info("Result directory: %s", output_dir)

    recordings = discover_recordings(data_dir)
    logging.info("Found %d recording(s) in %s", len(recordings), data_dir)
    if args.dry_run:
        for file_path in recordings:
            logging.info("[dry-run] Would process %s", file_path.name)
        logging.info("Dry run requested; skipping computation and persistence.")
        return
    if not recordings:
        logging.warning("No EEG recordings available; aborting.")
        return

    feature_flags = _select_feature_flags(args.features)
    bands = config.get("bands") or {}
    if not bands:
        logging.warning("No band definitions provided; power features disabled.")
    entropy_cfg = config.get("entropy") or {}
    entropy_bands = entropy_cfg.get("bands") or {}
    entropy_params = PermEntropyParams.from_mapping(entropy_cfg) if entropy_bands else None

    welch_params = WelchParams.from_mapping(config.get("welch"))
    abs_enabled = bool(bands) and (feature_flags["absolute_power"] or feature_flags["relative_power"])
    rel_enabled = bool(bands) and feature_flags["relative_power"]
    entropy_enabled = bool(entropy_bands) and feature_flags["permutation_entropy"]

    if not any([abs_enabled, rel_enabled, entropy_enabled]):
        raise ValueError("No features enabled after applying CLI and config constraints.")

    metadata_rows: List[Dict[str, float]] = []
    tidy_frames: List[pd.DataFrame] = []

    for file_path in recordings:
        subject_id = file_path.stem
        logging.info("Processing %s", file_path.name)
        raw = load_raw_file(file_path)
        metadata = summarise_recording(raw)
        metadata["subject_id"] = subject_id
        metadata_rows.append(metadata)

        abs_df = compute_absolute_power(raw, subject_id, bands, welch=welch_params) if abs_enabled else None
        rel_df = compute_relative_power(abs_df) if rel_enabled and abs_df is not None else None
        entropy_df = (
            compute_permutation_entropy(raw, subject_id, entropy_bands, params=entropy_params)
            if entropy_enabled
            else None
        )

        base_df = abs_df if abs_df is not None else EMPTY_FEATURE_FRAME.copy()
        tidy_frames.append(tidy_power_table(base_df, rel_df, entropy_df))

    tidy_df = pd.concat(tidy_frames, ignore_index=True) if tidy_frames else EMPTY_FEATURE_FRAME.copy()
    tidy_df.to_csv(paths["results_csv"], index=False)
    logging.info("Saved tidy dataset to %s (%d rows)", paths["results_csv"], len(tidy_df))

    qc_html = generate_qc_report(metadata_rows, tidy_df, config.get("report", {}))
    paths["qc_html"].write_text(qc_html, encoding="utf-8")
    logging.info("QC report ready: %s", paths["qc_html"])
    logging.info("qEEG pipeline finished.")


if __name__ == "__main__":
    main()
