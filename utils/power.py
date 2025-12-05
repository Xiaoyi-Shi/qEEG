"""Power-related EEG feature helpers used by CLI orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import mne
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WelchParams:
    """Container for PSD/Welch related parameters."""

    n_fft: int = 2048
    n_overlap: int = 1024
    n_per_seg: int | None = None

    @classmethod
    def from_mapping(cls, params: Mapping[str, int | None] | None) -> "WelchParams":
        """Create an instance from JSON/dict data."""
        if not params:
            return cls()
        return cls(
            n_fft=int(params.get("n_fft", cls.n_fft)),
            n_overlap=int(params.get("n_overlap", cls.n_overlap)),
            n_per_seg=(
                None if params.get("n_per_seg") in (None, "null") else int(params["n_per_seg"])
            ),
        )


def _bandpower(
    raw: mne.io.BaseRaw,
    fmin: float,
    fmax: float,
    *,
    picks: Sequence[str] | None = None,
    welch: WelchParams | None = None,
) -> pd.Series:
    """Compute band power for a frequency span."""
    params = welch or WelchParams()
    picks = picks or mne.pick_types(raw.info, eeg=True)
    try:
        spectrum = raw.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            picks=picks,
            n_fft=params.n_fft,
            n_overlap=params.n_overlap,
            n_per_seg=params.n_per_seg,
            verbose="ERROR",
        )
    except ValueError as exc:
        if "No frequencies found" in str(exc):
            LOGGER.warning(
                "Skipping PSD computation for band [%.2f, %.2f] Hz; no valid frequencies in recording.",
                fmin,
                fmax,
            )
            return pd.Series(dtype=float, name="power")
        raise
    psd, freqs = spectrum.get_data(return_freqs=True)
    df = np.trapz(psd, freqs, axis=1)
    df *= (1e6 ** 2)
    channel_names = np.array(raw.ch_names)[picks]
    return pd.Series(df, index=channel_names, name="power")


def _resolve_band_edges(raw: mne.io.BaseRaw, fmin: float | None, fmax: float | None) -> tuple[float, float] | None:
    """Clamp frequency bounds to the valid Nyquist range."""
    nyquist = float(raw.info["sfreq"]) / 2.0
    lower = 0.0 if fmin is None else max(0.0, float(fmin))
    upper = nyquist if fmax is None else min(float(fmax), nyquist - 1e-6)
    if upper <= lower:
        LOGGER.warning(
            "Skipping band [%s, %s]; valid frequency span must fall within (0, %.2f] Hz.",
            fmin,
            fmax,
            nyquist,
        )
        return None
    return lower, upper


def compute_absolute_power(
    raw: mne.io.BaseRaw,
    subject_id: str,
    bands: Mapping[str, Sequence[float]],
    *,
    picks: Sequence[str] | None = None,
    welch: Mapping[str, int] | WelchParams | None = None,
) -> pd.DataFrame:
    """Calculate absolute band power (μV^2/Hz) for the selected EEG channels."""
    welch_params = welch if isinstance(welch, WelchParams) else WelchParams.from_mapping(welch)
    raw_for_power = raw
    rows: list[dict[str, str | float]] = []
    for band, (fmin, fmax) in bands.items():
        resolved = _resolve_band_edges(raw_for_power, fmin, fmax)
        if resolved is None:
            continue
        adj_fmin, adj_fmax = resolved
        power_series = _bandpower(
            raw_for_power, adj_fmin, adj_fmax, picks=picks, welch=welch_params
        )
        if power_series.empty:
            continue
        for channel, power in power_series.items():
            rows.append(
                {
                    "subject_id": subject_id,
                    "channel": channel,
                    "band": band,
                    "metric": "absolute",
                    "power": float(power),
                }
            )

    return pd.DataFrame(rows)


def compute_relative_power(abs_power_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-channel relative power ratios from absolute power data."""
    required_cols = {"subject_id", "channel", "band", "power"}
    missing = required_cols - set(abs_power_df.columns)
    if missing:
        raise ValueError(f"Absolute power DataFrame is missing columns: {sorted(missing)}")

    grouped = abs_power_df.groupby(["subject_id", "channel"])["power"].transform("sum")
    rel_df = abs_power_df.copy()
    rel_df["power"] = rel_df["power"] / grouped
    rel_df["metric"] = "relative"
    return rel_df


def compute_power_ratios(
    abs_power_df: pd.DataFrame,
    ratio_map: Mapping[str, Sequence[str] | str] | None,
) -> pd.DataFrame:
    """Calculate ratios between named frequency bands using absolute power data."""
    if ratio_map is None or not ratio_map:
        return pd.DataFrame(columns=["subject_id", "channel", "band", "metric", "power"])
    required_cols = {"subject_id", "channel", "band", "power"}
    missing = required_cols - set(abs_power_df.columns)
    if missing:
        raise ValueError(f"Absolute power DataFrame is missing columns: {sorted(missing)}")
    if abs_power_df.empty:
        return pd.DataFrame(columns=list(abs_power_df.columns))

    pivot = (
        abs_power_df.pivot_table(
            index=["subject_id", "channel"],
            columns="band",
            values="power",
            aggfunc="first",
        )
        .sort_index()
    )
    records: list[dict[str, str | float]] = []
    for label, value in ratio_map.items():
        if isinstance(value, Sequence) and not isinstance(value, str):
            if len(value) != 2:
                raise ValueError(f"Ratio '{label}' must specify exactly two band names.")
            numerator, denominator = value
        else:
            numerator = label
            denominator = value
        numerator = str(numerator)
        denominator = str(denominator)
        if numerator not in pivot.columns or denominator not in pivot.columns:
            LOGGER.warning(
                "Skipping ratio %s (%s/%s); numerator or denominator band missing from absolute power.",
                label,
                numerator,
                denominator,
            )
            continue
        denom_series = pivot[denominator].replace(0.0, np.nan)
        ratio_series = pivot[numerator] / denom_series
        for (subject_id, channel), value in ratio_series.items():
            records.append(
                {
                    "subject_id": subject_id,
                    "channel": channel,
                    "band": str(label),
                    "metric": "ratio",
                    "power": float(value) if pd.notna(value) else np.nan,
                }
            )

    return pd.DataFrame(records, columns=["subject_id", "channel", "band", "metric", "power"])


def tidy_power_table(
    absolute_df: pd.DataFrame,
    *additional_frames: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return a concatenated tidy table ready for CSV export."""
    frames = [absolute_df]
    for frame in additional_frames:
        if frame is not None:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    ordered_cols = ["subject_id", "channel", "band", "metric", "power"]
    return result[ordered_cols]


__all__ = [
    "WelchParams",
    "compute_absolute_power",
    "compute_relative_power",
    "compute_power_ratios",
    "tidy_power_table",
]
