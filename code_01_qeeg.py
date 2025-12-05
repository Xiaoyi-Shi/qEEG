"""CLI entry point for qEEG processing tasks."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from utils.config import load_config, resolve_path
from utils.discovery import RecordingDescriptor, discover_bids_recordings, discover_recordings
from utils.entropy import (
    PermEntropyParams,
    SpectralEntropyParams,
    compute_permutation_entropy,
    compute_spectral_entropy,
)
from utils.power import (
    WelchParams,
    compute_absolute_power,
    compute_power_ratios,
    compute_relative_power,
    tidy_power_table,
)
from utils.preprocessing import SUPPORTED_EXTENSIONS, load_raw_file, preprocess_raw, summarise_recording
from utils.QC import generate_qc_report
from utils.runtime import configure_logging, ensure_output_tree
from utils.segment import SegmentConfig, compute_segment_rows, segment_rows_to_dataframe
SUPPORTED_FEATURES = {
    "absolute_power",
    "relative_power",
    "power_ratio",
    "permutation_entropy",
    "spectral_entropy",
}
FEATURE_COLUMNS = ["subject_id", "channel", "band", "metric", "power"]
EMPTY_FEATURE_FRAME = pd.DataFrame(columns=FEATURE_COLUMNS)
METRIC_LABELS = {
    "absolute": "absolute_power",
    "relative": "relative_power",
    "ratio": "power_ratio",
    "perm_entropy": "permutation_entropy",
    "spectral_entropy": "spectral_entropy",
}


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
        help="Override the flat EEG directory defined in the config file.",
    )
    parser.add_argument(
        "--bids-dir",
        type=Path,
        help="Override the BIDS root directory defined in the config file.",
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


def _select_feature_flags(requested: List[str] | None) -> Dict[str, bool]:
    """Normalize CLI feature selections into boolean flags."""
    if not requested:
        return {name: True for name in SUPPORTED_FEATURES}
    flags = {name: name in requested for name in SUPPORTED_FEATURES}
    if flags["relative_power"]:
        flags["absolute_power"] = True
    if flags["power_ratio"]:
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
    preprocess_cfg = config.get("preprocessing")
    segment_cfg = SegmentConfig.from_mapping(config.get("Segment") or config.get("segment"))
    output_root = args.result_dir or resolve_path(paths_cfg.get("output_dir", "result"), base_dir)

    data_dir = args.data_dir
    bids_dir = args.bids_dir
    if bids_dir is None and data_dir is None:
        bids_cfg = paths_cfg.get("bids_dir")
        if bids_cfg:
            bids_dir = resolve_path(bids_cfg, base_dir)
    if bids_dir is None and data_dir is None:
        data_dir = resolve_path(paths_cfg.get("data_dir", "data/EEG_DATA"), base_dir)
    elif data_dir is not None:
        data_dir = data_dir.resolve()
    if bids_dir is not None:
        bids_dir = bids_dir.resolve()
    if data_dir is not None and bids_dir is not None:
        raise ValueError("Provide either a flat data directory or a BIDS directory, not both.")
    input_label = "BIDS" if bids_dir else "flat"

    current_time = datetime.now()
    output_dir = output_root / current_time.strftime("%Y-%m-%d-%H-%M")
    paths = ensure_output_tree(output_dir)
    configure_logging(paths["log_file"], args.log_level)

    logging.info("Starting qEEG pipeline (config=%s)", config_path)
    if bids_dir:
        logging.info("BIDS directory: %s", bids_dir)
    else:
        logging.info("Data directory: %s", data_dir)
    logging.info("Result directory: %s", output_dir)
    if segment_cfg.enabled:
        logging.info(
            "Segmented processing enabled (segment_length=%.2f s, bad_tolerance=%.2f)",
            segment_cfg.segment_length,
            segment_cfg.bad_tolerance,
        )

    if bids_dir:
        recordings = discover_bids_recordings(bids_dir)
    else:
        if data_dir is None:
            raise ValueError("Data directory is undefined.")
        recordings = discover_recordings(data_dir)
    logging.info("Found %d recording(s) via %s input", len(recordings), input_label)
    if args.dry_run:
        for descriptor in recordings:
            logging.info("[dry-run] Would process %s (%s)", descriptor.path.name, descriptor.subject_id)
        logging.info("Dry run requested; skipping computation and persistence.")
        return
    if not recordings:
        logging.warning("No EEG recordings available; aborting.")
        return

    feature_flags = _select_feature_flags(args.features)
    power_cfg = config.get("power") or {}
    bands = power_cfg.get("bands") or {}
    if not bands:
        logging.warning("No band definitions provided; power features disabled.")
    ratio_bands = power_cfg.get("ratio_bands") or {}
    if feature_flags["power_ratio"] and not ratio_bands:
        logging.warning("Power ratio feature requested but 'ratio_bands' is empty; disabling ratios.")
    welch_params = WelchParams.from_mapping(power_cfg.get("welch"))

    entropy_cfg = config.get("entropy") or {}
    perm_entropy_cfg = entropy_cfg.get("permutation") or {}
    entropy_bands = perm_entropy_cfg.get("bands") or {}
    entropy_params = PermEntropyParams.from_mapping(perm_entropy_cfg) if entropy_bands else None

    spectral_cfg = entropy_cfg.get("spectral") or {}
    spectral_params = SpectralEntropyParams.from_mapping(spectral_cfg) if spectral_cfg else None
    spectral_band_label = spectral_cfg.get("band_label") if spectral_cfg else None
    ratio_enabled = bool(ratio_bands) and feature_flags["power_ratio"]
    abs_enabled = bool(bands) and (
        feature_flags["absolute_power"] or feature_flags["relative_power"] or ratio_enabled
    )
    rel_enabled = bool(bands) and feature_flags["relative_power"]
    entropy_enabled = bool(entropy_bands) and feature_flags["permutation_entropy"]
    spectral_enabled = bool(spectral_cfg) and feature_flags["spectral_entropy"]

    if not any([abs_enabled, rel_enabled, ratio_enabled, entropy_enabled, spectral_enabled]):
        raise ValueError("No features enabled after applying CLI and config constraints.")

    metadata_rows: List[Dict[str, float]] = []
    tidy_frames: List[pd.DataFrame] = []
    segment_rows: List[Dict[str, Any]] = []

    for descriptor in recordings:
        subject_id = descriptor.subject_id
        logging.info("Processing %s", descriptor.path.name)
        raw = load_raw_file(descriptor.path)
        raw = preprocess_raw(raw, preprocess_cfg, base_dir=base_dir)
        metadata = summarise_recording(raw)
        metadata["subject_id"] = subject_id
        metadata_rows.append(metadata)

        abs_df = compute_absolute_power(raw, subject_id, bands, welch=welch_params) if abs_enabled else None
        rel_df = compute_relative_power(abs_df) if rel_enabled and abs_df is not None else None
        ratio_df = (
            compute_power_ratios(abs_df, ratio_bands) if ratio_enabled and abs_df is not None else None
        )
        entropy_df = (
            compute_permutation_entropy(raw, subject_id, entropy_bands, params=entropy_params)
            if entropy_enabled
            else None
        )
        spectral_df = (
            compute_spectral_entropy(
                raw,
                subject_id,
                params=spectral_params,
                band_label=spectral_band_label,
            )
            if spectral_enabled
            else None
        )

        base_df = abs_df if abs_df is not None else EMPTY_FEATURE_FRAME.copy()
        tidy_frames.append(tidy_power_table(base_df, rel_df, ratio_df, entropy_df, spectral_df))

        if segment_cfg.enabled:
            segment_rows.extend(
                compute_segment_rows(
                    raw,
                    subject_id,
                    segment_cfg=segment_cfg,
                    bands=bands,
                    ratio_bands=ratio_bands,
                    welch_params=welch_params,
                    entropy_bands=entropy_bands,
                    entropy_params=entropy_params,
                    spectral_params=spectral_params,
                    spectral_band_label=spectral_band_label,
                    abs_enabled=abs_enabled,
                    rel_enabled=rel_enabled,
                    ratio_enabled=ratio_enabled,
                    entropy_enabled=entropy_enabled,
                    spectral_enabled=spectral_enabled,
                    metric_labels=METRIC_LABELS,
                )
            )

    tidy_df = pd.concat(tidy_frames, ignore_index=True) if tidy_frames else EMPTY_FEATURE_FRAME.copy()
    tidy_df.to_csv(paths["results_csv"], index=False)
    logging.info("Saved tidy dataset to %s (%d rows)", paths["results_csv"], len(tidy_df))

    segment_df: pd.DataFrame | None = None
    if segment_cfg.enabled:
        segment_df = segment_rows_to_dataframe(segment_rows)

    qc_html = generate_qc_report(
        metadata_rows,
        tidy_df,
        config.get("report", {}),
        segment_df=segment_df,
    )
    paths["qc_html"].write_text(qc_html, encoding="utf-8")
    logging.info("QC report ready: %s", paths["qc_html"])

    if segment_df is not None:
        if segment_df.empty:
            logging.info("Segmented output enabled but no rows were produced; skipping CSV export.")
        else:
            segment_df.to_csv(paths["segment_csv"], index=False)
            logging.info("Saved segmented dataset to %s (%d rows)", paths["segment_csv"], len(segment_df))
    logging.info("qEEG pipeline finished.")


if __name__ == "__main__":
    main()
