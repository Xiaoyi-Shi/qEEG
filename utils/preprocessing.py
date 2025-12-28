"""EEG loading and preprocessing helpers shared by CLI orchestration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import mne

SUPPORTED_EXTENSIONS = (".fif", ".edf")
LOGGER = logging.getLogger(__name__)


def load_raw_file(file_path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    """Load an EEG recording using the appropriate MNE reader based on file suffix."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported EEG format '{path.suffix}'. Expected {SUPPORTED_EXTENSIONS}."
        )

    if path.suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=preload, verbose="ERROR")
    else:
        raw = mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    return raw


def preprocess_raw(
    raw: mne.io.BaseRaw,
    settings: Mapping[str, Any] | None,
    *,
    base_dir: Path | None = None,
) -> mne.io.BaseRaw:
    """Apply optional preprocessing (channel picks, montage, filtering, resampling, reference)."""
    if not settings:
        settings = {}

    channel_cfg = settings.get("channel")
    if channel_cfg:
        _apply_channel_selection(raw, channel_cfg)

    montage_cfg = settings.get("montage")
    if montage_cfg:
        _apply_montage(raw, montage_cfg, base_dir=base_dir)

    notch_cfg = settings.get("notch")
    if notch_cfg:
        _apply_notch_filter(raw, notch_cfg)

    bandpass_cfg = settings.get("bandpass")
    if bandpass_cfg:
        _apply_bandpass_filter(raw, bandpass_cfg)

    resample_target = settings.get("resample_hz") or settings.get("resample")
    if resample_target:
        _apply_resample(raw, resample_target)

    reference_cfg = settings.get("reference")
    if reference_cfg:
        _apply_reference(raw, reference_cfg)
    else:
        LOGGER.info("No reference configuration provided; keeping existing reference.")
    return raw


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


def _apply_montage(
    raw: mne.io.BaseRaw,
    montage_cfg: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> None:
    """Attach a montage using either a built-in name or a custom file path."""
    if raw.get_montage() is not None or raw.info.get("dig"):
        LOGGER.info("Montage already present on recording; skipping configured montage.")
        return
    name = montage_cfg.get("name") or montage_cfg.get("kind")
    path_value = montage_cfg.get("path") or montage_cfg.get("filepath")
    match_case = bool(montage_cfg.get("match_case", False))
    if path_value:
        montage_path = Path(path_value)
        if not montage_path.is_absolute():
            montage_path = (
                (base_dir / montage_path).resolve() if base_dir else montage_path.resolve()
            )
        if not montage_path.exists():
            raise FileNotFoundError(f"Montage file not found: {montage_path}")
        montage = mne.channels.read_custom_montage(montage_path)
        descriptor = str(montage_path)
    elif name:
        montage = mne.channels.make_standard_montage(str(name))
        descriptor = str(name)
    else:
        raise ValueError("Montage configuration requires either 'name' or 'path'.")
    raw.set_montage(montage, match_case=match_case, verbose="ERROR")
    LOGGER.info("Applied montage: %s", descriptor)


def _apply_notch_filter(raw: mne.io.BaseRaw, notch_cfg: Mapping[str, Any]) -> None:
    """Apply a notch filter based on the provided configuration."""
    freqs = notch_cfg.get("freqs") or notch_cfg.get("frequency")
    if freqs is None:
        raise ValueError("Notch filter configuration requires 'freqs'.")
    if isinstance(freqs, (int, float)):
        freq_values = [float(freqs)]
    else:
        freq_values = [float(value) for value in freqs]
    kwargs = {key: value for key, value in notch_cfg.items() if key not in {"freqs", "frequency"}}
    raw.notch_filter(freqs=freq_values, verbose="ERROR", **kwargs)
    LOGGER.info("Applied notch filter at %s Hz", freq_values)


def _apply_bandpass_filter(raw: mne.io.BaseRaw, bandpass_cfg: Mapping[str, Any]) -> None:
    """Apply a bandpass/low/high-pass filter."""
    l_freq = bandpass_cfg.get("l_freq")
    h_freq = bandpass_cfg.get("h_freq")
    if l_freq is None and h_freq is None:
        raise ValueError("Bandpass configuration requires 'l_freq' or 'h_freq'.")
    kwargs = {key: value for key, value in bandpass_cfg.items() if key not in {"l_freq", "h_freq"}}
    raw.filter(
        l_freq=None if l_freq is None else float(l_freq),
        h_freq=None if h_freq is None else float(h_freq),
        verbose="ERROR",
        **kwargs,
    )
    LOGGER.info("Applied bandpass filter l_freq=%s h_freq=%s", l_freq, h_freq)


def _apply_resample(raw: mne.io.BaseRaw, target: Any) -> None:
    """Resample the recording."""
    try:
        sfreq = float(target)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid resample target '{target}'.") from exc
    if sfreq <= 0:
        raise ValueError("Resample frequency must be positive.")
    raw.resample(sfreq=sfreq, npad="auto", verbose="ERROR")
    LOGGER.info("Resampled recording to %.2f Hz", sfreq)


def _apply_reference(raw: mne.io.BaseRaw, reference_cfg: Mapping[str, Any]) -> None:
    """Apply the requested reference strategy."""
    ref_kind = str(reference_cfg.get("kind", "average")).lower()
    projection = bool(reference_cfg.get("projection", False))
    if ref_kind in {"none", "off", "disable"}:
        LOGGER.info("Skipping EEG re-referencing per configuration.")
        return
    if ref_kind in {"average", "avg"}:
        raw.set_eeg_reference("average", projection=projection, verbose="ERROR")
        LOGGER.info("Applied average EEG reference.")
        return
    if ref_kind in {"channel", "channels", "custom"}:
        channels = reference_cfg.get("channels")
        if isinstance(channels, str):
            channels = [channels]
        if not channels:
            raise ValueError("Channel reference requires a non-empty 'channels' list.")
        raw.set_eeg_reference(ref_channels=list(channels), projection=projection, verbose="ERROR")
        LOGGER.info("Applied channel reference using: %s", ", ".join(channels))
        return
    raise ValueError(f"Unknown reference kind '{ref_kind}'.")


def _normalize_channel_list(value: Any) -> list[str] | None:
    """Coerce a channel selector into a list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("Channel selections must be a string or sequence of strings.")


def _apply_channel_selection(raw: mne.io.BaseRaw, channel_cfg: Mapping[str, Any]) -> None:
    """Apply channel inclusion/exclusion rules."""
    select = _normalize_channel_list(channel_cfg.get("select"))
    exclude = _normalize_channel_list(channel_cfg.get("del"))
    if not select and not exclude:
        LOGGER.info("Channel selection block provided but empty; retaining all channels.")
        return

    current = list(raw.ch_names)
    picks = select or current
    missing_select = [name for name in picks if name not in current]
    if missing_select:
        picks = [name for name in picks if name in current]
        LOGGER.warning(
            "Selected channels not found and will be ignored: %s", ", ".join(missing_select)
        )

    exclude_present = [name for name in exclude or [] if name in current]
    missing_exclude = [name for name in exclude or [] if name not in current]
    if missing_exclude:
        LOGGER.warning(
            "Exclusion channels not found and will be ignored: %s", ", ".join(missing_exclude)
        )

    if not picks:
        raise ValueError("No channels remain after applying selection rules.")

    final_picks = [name for name in picks if name not in set(exclude_present)]
    if not final_picks:
        raise ValueError("Exclusion list removed all selected channels.")

    raw.pick(picks=final_picks, exclude=exclude_present)
    LOGGER.info(
        "Applied channel selection: kept %d of %d channels (excluded %d).",
        len(raw.ch_names),
        len(current),
        len(exclude_present),
    )


__all__ = ["SUPPORTED_EXTENSIONS", "load_raw_file", "preprocess_raw", "summarise_recording"]
