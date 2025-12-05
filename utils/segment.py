"""Segmentation helpers shared by CLI orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import mne
import pandas as pd

from utils.power import WelchParams, compute_absolute_power, compute_power_ratios, compute_relative_power
from utils.entropy import (
    PermEntropyParams,
    SpectralEntropyParams,
    compute_permutation_entropy,
    compute_spectral_entropy,
)

NAN = float("nan")


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


def _accumulate_segment_frame(
    frame: pd.DataFrame | None,
    *,
    row_map: MutableMapping[tuple[str, str, str], List[float]],
    segment_index: int,
    total_segments: int,
    metric_labels: Mapping[str, str],
) -> None:
    """Store per-segment feature values into the aggregation map."""
    if frame is None or frame.empty:
        return
    for _, row in frame.iterrows():
        metric_label = metric_labels.get(str(row["metric"]), str(row["metric"]))
        band_value = row.get("band")
        band = None if band_value is None or pd.isna(band_value) else band_value
        entity = f"{metric_label}[{band}]" if band not in (None, "") else metric_label
        key = (row["subject_id"], entity, row["channel"])
        bucket = row_map.setdefault(key, [NAN] * total_segments)
        bucket[segment_index] = float(row["power"])


def compute_segment_rows(
    raw: "mne.io.BaseRaw",
    subject_id: str,
    *,
    segment_cfg: SegmentConfig,
    bands: Dict[str, Sequence[float]],
    ratio_bands: Mapping[str, Sequence[str] | str],
    welch_params: WelchParams,
    entropy_bands: Dict[str, Sequence[float]],
    entropy_params: PermEntropyParams | None,
    spectral_params: SpectralEntropyParams | None,
    spectral_band_label: str | None,
    abs_enabled: bool,
    rel_enabled: bool,
    ratio_enabled: bool,
    entropy_enabled: bool,
    spectral_enabled: bool,
    metric_labels: Mapping[str, str],
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
        ratio_df = (
            compute_power_ratios(abs_df, ratio_bands) if ratio_enabled and abs_df is not None else None
        )
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
        _accumulate_segment_frame(
            abs_df,
            row_map=row_map,
            segment_index=window.index,
            total_segments=len(windows),
            metric_labels=metric_labels,
        )
        _accumulate_segment_frame(
            rel_df,
            row_map=row_map,
            segment_index=window.index,
            total_segments=len(windows),
            metric_labels=metric_labels,
        )
        _accumulate_segment_frame(
            ratio_df,
            row_map=row_map,
            segment_index=window.index,
            total_segments=len(windows),
            metric_labels=metric_labels,
        )
        _accumulate_segment_frame(
            entropy_df,
            row_map=row_map,
            segment_index=window.index,
            total_segments=len(windows),
            metric_labels=metric_labels,
        )
        _accumulate_segment_frame(
            spectral_df,
            row_map=row_map,
            segment_index=window.index,
            total_segments=len(windows),
            metric_labels=metric_labels,
        )

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


def segment_rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
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


__all__ = ["SegmentConfig", "compute_segment_rows", "segment_rows_to_dataframe"]
