"""CLI entry point for qEEG processing tasks."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

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
METRIC_LABELS = {
    "absolute": "absolute_power",
    "relative": "relative_power",
    "perm_entropy": "permutation_entropy",
    "spectral_entropy": "spectral_entropy",
}
NAN = float("nan")


@dataclass(frozen=True)
class RecordingDescriptor:
    """Normalized structure describing an EEG file slated for processing."""

    path: Path
    subject_id: str


@dataclass(frozen=True)
class SegmentConfig:
    """Runtime configuration for segmented feature calculations."""

    segment_length: float | None = None
    bad_tolerance: float = 0.5

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None) -> "SegmentConfig":
        if not params:
            return cls()
        raw_length = params.get("Segment_length", params.get("segment_length"))
        if isinstance(raw_length, str):
            normalized = raw_length.strip().lower()
            if normalized in {"", "none", "null"}:
                raw_length = None
        if raw_length is not None:
            try:
                segment_length = float(raw_length)
            except (TypeError, ValueError) as exc:
                raise ValueError("Segment_length must be numeric or null.") from exc
            if segment_length <= 0:
                raise ValueError("Segment_length must be positive when provided.")
        else:
            segment_length = None

        tolerance_value = params.get("bad_segment_tolerance", params.get("bad_tolerance"))
        if tolerance_value is None:
            bad_tolerance = cls.bad_tolerance
        else:
            try:
                bad_tolerance = float(tolerance_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("bad_segment_tolerance must be numeric.") from exc
            if not 0.0 <= bad_tolerance <= 1.0:
                raise ValueError("bad_segment_tolerance must be within [0, 1].")
        return cls(segment_length=segment_length, bad_tolerance=bad_tolerance)

    @property
    def enabled(self) -> bool:
        return self.segment_length is not None


@dataclass(frozen=True)
class SegmentWindow:
    """Container describing an individual time range for segmentation."""

    index: int
    tmin: float
    tmax: float
    is_bad: bool


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
        "segment_csv": output_dir / "qEEG_segment_result.csv",
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


def _build_segment_windows(raw: "mne.io.BaseRaw", config: SegmentConfig) -> List[SegmentWindow]:
    """Return segmentation windows along the recording."""
    if not config.enabled:
        return []
    duration = float(raw.n_times) / float(raw.info["sfreq"])
    if duration <= 0 or config.segment_length is None:
        return []
    windows: List[SegmentWindow] = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(start + config.segment_length, duration)
        bad_fraction = _bad_coverage_fraction(raw, start, end)
        windows.append(SegmentWindow(index=index, tmin=start, tmax=end, is_bad=bad_fraction > config.bad_tolerance))
        start = end
        index += 1
    return windows


def _bad_coverage_fraction(raw: "mne.io.BaseRaw", start: float, end: float) -> float:
    """Return the proportion of a window that overlaps 'bad' annotations."""
    annotations = getattr(raw, "annotations", None)
    if annotations is None or len(annotations) == 0:
        return 0.0
    coverage = 0.0
    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        if "bad" not in str(desc).lower():
            continue
        ann_start = float(onset)
        ann_end = float(onset + duration)
        overlap = max(0.0, min(end, ann_end) - max(start, ann_start))
        coverage += overlap
    span = end - start
    if span <= 0:
        return 0.0
    return min(coverage / span, 1.0)


def _accumulate_segment_frame(
    frame: pd.DataFrame | None,
    *,
    row_map: Dict[tuple[str, str, str], List[float]],
    segment_index: int,
    total_segments: int,
) -> None:
    """Store per-segment feature values into the aggregation map."""
    if frame is None or frame.empty:
        return
    for _, row in frame.iterrows():
        metric_label = METRIC_LABELS.get(str(row["metric"]), str(row["metric"]))
        band_value = row.get("band")
        band = None if band_value is None or pd.isna(band_value) else band_value
        entity = f"{metric_label}[{band}]" if band not in (None, "") else metric_label
        key = (row["subject_id"], entity, row["channel"])
        bucket = row_map.setdefault(key, [NAN] * total_segments)
        bucket[segment_index] = float(row["power"])


def _compute_segment_rows(
    raw: "mne.io.BaseRaw",
    subject_id: str,
    *,
    segment_cfg: SegmentConfig,
    bands: Dict[str, List[float]],
    welch_params: WelchParams,
    entropy_bands: Dict[str, List[float]],
    entropy_params: PermEntropyParams | None,
    spectral_params: SpectralEntropyParams | None,
    spectral_band_label: str | None,
    abs_enabled: bool,
    rel_enabled: bool,
    entropy_enabled: bool,
    spectral_enabled: bool,
) -> List[Dict[str, Any]]:
    """Compute segment-level feature rows for the configured entities."""
    if not segment_cfg.enabled:
        return []
    windows = _build_segment_windows(raw, segment_cfg)
    if not windows:
        return []
    row_map: Dict[tuple[str, str, str], List[float]] = {}
    for window in windows:
        if window.is_bad:
            continue
        segment_raw = raw.copy().crop(tmin=window.tmin, tmax=window.tmax, include_tmax=False)
        abs_df = compute_absolute_power(segment_raw, subject_id, bands, welch=welch_params) if abs_enabled else None
        rel_df = compute_relative_power(abs_df) if rel_enabled and abs_df is not None else None
        entropy_df = (
            compute_permutation_entropy(segment_raw, subject_id, entropy_bands, params=entropy_params)
            if entropy_enabled
            else None
        )
        spectral_df = (
            compute_spectral_entropy(
                segment_raw,
                subject_id,
                params=spectral_params,
                band_label=spectral_band_label,
            )
            if spectral_enabled
            else None
        )
        _accumulate_segment_frame(abs_df, row_map=row_map, segment_index=window.index, total_segments=len(windows))
        _accumulate_segment_frame(rel_df, row_map=row_map, segment_index=window.index, total_segments=len(windows))
        _accumulate_segment_frame(entropy_df, row_map=row_map, segment_index=window.index, total_segments=len(windows))
        _accumulate_segment_frame(spectral_df, row_map=row_map, segment_index=window.index, total_segments=len(windows))

    rows: List[Dict[str, Any]] = []
    for (subj, entity, channel), values in row_map.items():
        rows.append(
            {
                "subject_id": subj,
                "entity": entity,
                "channel": channel,
                "values": values,
            }
        )
    return rows


def _segment_rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert aggregated segment rows into a wide DataFrame."""
    if not rows:
        return pd.DataFrame(columns=["subject_id", "entity", "channel"])
    max_segments = max(len(row["values"]) for row in rows)
    segment_columns = [f"segment_{idx+1:03d}" for idx in range(max_segments)]
    records: List[Dict[str, Any]] = []
    for row in rows:
        padded = list(row["values"]) + [NAN] * (max_segments - len(row["values"]))
        record = {
            "subject_id": row["subject_id"],
            "entity": row["entity"],
            "channel": row["channel"],
        }
        record.update({column: padded[idx] for idx, column in enumerate(segment_columns)})
        records.append(record)
    return pd.DataFrame(records, columns=["subject_id", "entity", "channel", *segment_columns])


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
    welch_params = WelchParams.from_mapping(power_cfg.get("welch"))

    entropy_cfg = config.get("entropy") or {}
    perm_entropy_cfg = entropy_cfg.get("permutation") or {}
    entropy_bands = perm_entropy_cfg.get("bands") or {}
    entropy_params = PermEntropyParams.from_mapping(perm_entropy_cfg) if entropy_bands else None

    spectral_cfg = entropy_cfg.get("spectral") or {}
    spectral_params = SpectralEntropyParams.from_mapping(spectral_cfg) if spectral_cfg else None
    spectral_band_label = spectral_cfg.get("band_label") if spectral_cfg else None
    abs_enabled = bool(bands) and (feature_flags["absolute_power"] or feature_flags["relative_power"])
    rel_enabled = bool(bands) and feature_flags["relative_power"]
    entropy_enabled = bool(entropy_bands) and feature_flags["permutation_entropy"]
    spectral_enabled = bool(spectral_cfg) and feature_flags["spectral_entropy"]

    if not any([abs_enabled, rel_enabled, entropy_enabled, spectral_enabled]):
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

        if segment_cfg.enabled:
            segment_rows.extend(
                _compute_segment_rows(
                    raw,
                    subject_id,
                    segment_cfg=segment_cfg,
                    bands=bands,
                    welch_params=welch_params,
                    entropy_bands=entropy_bands,
                    entropy_params=entropy_params,
                    spectral_params=spectral_params,
                    spectral_band_label=spectral_band_label,
                    abs_enabled=abs_enabled,
                    rel_enabled=rel_enabled,
                    entropy_enabled=entropy_enabled,
                    spectral_enabled=spectral_enabled,
                )
            )

    tidy_df = pd.concat(tidy_frames, ignore_index=True) if tidy_frames else EMPTY_FEATURE_FRAME.copy()
    tidy_df.to_csv(paths["results_csv"], index=False)
    logging.info("Saved tidy dataset to %s (%d rows)", paths["results_csv"], len(tidy_df))

    qc_html = generate_qc_report(metadata_rows, tidy_df, config.get("report", {}))
    paths["qc_html"].write_text(qc_html, encoding="utf-8")
    logging.info("QC report ready: %s", paths["qc_html"])

    if segment_cfg.enabled:
        segment_df = _segment_rows_to_dataframe(segment_rows)
        if segment_df.empty:
            logging.info("Segmented output enabled but no rows were produced; skipping CSV export.")
        else:
            segment_df.to_csv(paths["segment_csv"], index=False)
            logging.info("Saved segmented dataset to %s (%d rows)", paths["segment_csv"], len(segment_df))
    logging.info("qEEG pipeline finished.")


if __name__ == "__main__":
    main()
