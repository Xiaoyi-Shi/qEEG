"""CLI entry point for qEEG processing tasks."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from utils.basefun import (
    SUPPORTED_EXTENSIONS,
    WelchParams,
    compute_absolute_power,
    compute_relative_power,
    load_raw_file,
    preprocess_raw,
    summarise_recording,
    tidy_power_table,
)
from utils.entropy import (
    PermEntropyParams,
    SpectralEntropyParams,
    compute_permutation_entropy,
    compute_spectral_entropy,
)
from utils.QC import generate_qc_report

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
SUPPORTED_FEATURES = {"absolute_power", "relative_power", "permutation_entropy", "spectral_entropy"}
FEATURE_COLUMNS = ["subject_id", "channel", "band", "metric", "power"]
EMPTY_FEATURE_FRAME = pd.DataFrame(columns=FEATURE_COLUMNS)


@dataclass(frozen=True)
class RecordingDescriptor:
    """Normalized structure describing an EEG file slated for processing."""

    path: Path
    subject_id: str


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


def discover_recordings(data_dir: Path) -> List[RecordingDescriptor]:
    """Return descriptors for supported EEG recording files in a flat directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files: List[Path] = []
    for suffix in SUPPORTED_EXTENSIONS:
        files.extend(sorted(data_dir.glob(f"*{suffix}")))
    descriptors = [
        RecordingDescriptor(path=file_path, subject_id=_derive_subject_id(file_path)) for file_path in files
    ]
    return descriptors


def discover_bids_recordings(bids_root: Path) -> List[RecordingDescriptor]:
    """Return descriptors for EEG files stored in a BIDS dataset."""
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS directory not found: {bids_root}")
    subject_dirs = [path for path in bids_root.iterdir() if path.is_dir() and path.name.startswith("sub-")]
    if not subject_dirs:
        raise FileNotFoundError(f"No BIDS subject directories located under: {bids_root}")

    eeg_files: List[Path] = []
    for suffix in SUPPORTED_EXTENSIONS:
        eeg_files.extend(
            sorted(
                bids_root.rglob(f"*{suffix}"),
            )
        )
    filtered = [path for path in eeg_files if path.parent.name.lower() == "eeg"]
    descriptors = [
        RecordingDescriptor(path=file_path, subject_id=_derive_subject_id(file_path, trim_bids_suffix=True))
        for file_path in filtered
    ]
    return descriptors


def _derive_subject_id(file_path: Path, *, trim_bids_suffix: bool = False) -> str:
    """Normalize subject IDs derived from EEG filenames."""
    stem = file_path.stem
    if trim_bids_suffix and stem.endswith("_eeg"):
        stem = stem[: -len("_eeg")]
    return stem


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
    preprocess_cfg = config.get("preprocessing")
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
    bands = config.get("bands") or {}
    if not bands:
        logging.warning("No band definitions provided; power features disabled.")
    entropy_cfg = config.get("entropy") or {}
    entropy_bands = entropy_cfg.get("bands") or {}
    entropy_params = PermEntropyParams.from_mapping(entropy_cfg) if entropy_bands else None

    spectral_cfg = config.get("spectral_entropy") or {}
    spectral_params = SpectralEntropyParams.from_mapping(spectral_cfg) if spectral_cfg else None
    spectral_band_label = spectral_cfg.get("band_label") if spectral_cfg else None

    welch_params = WelchParams.from_mapping(config.get("welch"))
    abs_enabled = bool(bands) and (feature_flags["absolute_power"] or feature_flags["relative_power"])
    rel_enabled = bool(bands) and feature_flags["relative_power"]
    entropy_enabled = bool(entropy_bands) and feature_flags["permutation_entropy"]
    spectral_enabled = bool(spectral_cfg) and feature_flags["spectral_entropy"]

    if not any([abs_enabled, rel_enabled, entropy_enabled, spectral_enabled]):
        raise ValueError("No features enabled after applying CLI and config constraints.")

    metadata_rows: List[Dict[str, float]] = []
    tidy_frames: List[pd.DataFrame] = []

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
        tidy_frames.append(tidy_power_table(base_df, rel_df, entropy_df, spectral_df))

    tidy_df = pd.concat(tidy_frames, ignore_index=True) if tidy_frames else EMPTY_FEATURE_FRAME.copy()
    tidy_df.to_csv(paths["results_csv"], index=False)
    logging.info("Saved tidy dataset to %s (%d rows)", paths["results_csv"], len(tidy_df))

    qc_html = generate_qc_report(metadata_rows, tidy_df, config.get("report", {}))
    paths["qc_html"].write_text(qc_html, encoding="utf-8")
    logging.info("QC report ready: %s", paths["qc_html"])
    logging.info("qEEG pipeline finished.")


if __name__ == "__main__":
    main()
