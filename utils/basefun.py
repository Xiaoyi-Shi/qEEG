"""Power-related EEG feature helpers used by CLI orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import mne
import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = (".fif", ".edf")
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


def load_raw_file(file_path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load an EEG recording using the appropriate MNE reader based on file suffix."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported EEG format '{path.suffix}'. Expected {SUPPORTED_EXTENSIONS}.")

    if path.suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    else:
        raw = mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    return raw


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


def _apply_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Return a copy of the recording that uses a common average reference."""
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if eeg_picks.size == 0:
        return raw
    referenced = raw.copy()
    referenced.set_eeg_reference("average", projection=False, verbose="ERROR")
    return referenced


def compute_absolute_power(
    raw: mne.io.BaseRaw,
    subject_id: str,
    bands: Mapping[str, Sequence[float]],
    *,
    picks: Sequence[str] | None = None,
    welch: Mapping[str, int] | WelchParams | None = None,
) -> pd.DataFrame:
    """Calculate absolute band power (µV^2/Hz) for the selected EEG channels."""
    welch_params = welch if isinstance(welch, WelchParams) else WelchParams.from_mapping(welch)
    raw_for_power = _apply_average_reference(raw)
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


def summarise_recording(raw: mne.io.BaseRaw) -> Dict[str, str | int | float]:
    """Extract a handful of metadata fields for QC reporting."""
    info = raw.info
    duration = raw.n_times / info["sfreq"]
    return {
        "n_channels": info["nchan"],
        "sfreq": float(info["sfreq"]),
        "duration_sec": float(duration),
        "highpass": info.get("highpass"),
        "lowpass": info.get("lowpass"),
    }
