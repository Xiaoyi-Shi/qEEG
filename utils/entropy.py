"""Entropy-based EEG feature helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import antropy as ant
import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PermEntropyParams:
    """Container for permutation entropy hyper-parameters."""

    order: int = 3
    delay: int = 1
    normalize: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, object] | None) -> "PermEntropyParams":
        if not params:
            return cls()
        return cls(
            order=int(params.get("order", cls.order)),
            delay=int(params.get("delay", cls.delay)),
            normalize=bool(params.get("normalize", cls.normalize)),
        )


def _validate_bands(bands: Mapping[str, Sequence[float]]) -> dict[str, tuple[float | None, float | None]]:
    normalized: dict[str, tuple[float | None, float | None]] = {}
    for band, limits in bands.items():
        if not isinstance(limits, Sequence) or len(limits) != 2:
            raise ValueError(f"Band '{band}' must define [fmin, fmax].")
        fmin, fmax = limits
        normalized[band] = (
            None if fmin is None else float(fmin),
            None if fmax is None else float(fmax),
        )
    return normalized


def compute_permutation_entropy(
    raw: mne.io.BaseRaw,
    subject_id: str,
    bands: Mapping[str, Sequence[float]],
    *,
    params: PermEntropyParams | Mapping[str, object] | None = None,
    picks: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Calculate permutation entropy for all requested frequency bands and channels."""
    if not bands:
        raise ValueError("Permutation entropy requires at least one band definition.")

    entropy_params = (
        params if isinstance(params, PermEntropyParams) else PermEntropyParams.from_mapping(params)
    )
    picks = picks or mne.pick_types(raw.info, eeg=True)
    if len(picks) == 0:
        raise ValueError("No EEG channels available for permutation entropy computation.")

    normalized_bands = _validate_bands(bands)
    base_data = raw.get_data(picks=picks)
    channel_names = np.array(raw.ch_names)[picks]
    rows: list[dict[str, str | float]] = []

    for band, (fmin, fmax) in normalized_bands.items():
        band_data = mne.filter.filter_data(
            base_data.copy(),
            sfreq=raw.info["sfreq"],
            l_freq=fmin,
            h_freq=fmax,
            verbose="ERROR",
        )
        for channel_idx, channel in enumerate(channel_names):
            entropy_value = ant.perm_entropy(
                band_data[channel_idx],
                order=entropy_params.order,
                delay=entropy_params.delay,
                normalize=entropy_params.normalize,
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "channel": channel,
                    "band": band,
                    "metric": "perm_entropy",
                    "power": float(entropy_value),
                }
            )

    return pd.DataFrame(rows)
