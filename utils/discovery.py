"""Recording discovery helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from utils.preprocessing import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class RecordingDescriptor:
    """Normalized structure describing an EEG file slated for processing."""

    path: Path
    subject_id: str


def _derive_subject_id(file_path: Path, *, trim_bids_suffix: bool = False) -> str:
    """Normalize subject IDs derived from EEG filenames."""
    stem = file_path.stem
    if trim_bids_suffix and stem.endswith("_eeg"):
        stem = stem[: -len("_eeg")]
    return stem


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


__all__ = ["RecordingDescriptor", "discover_recordings", "discover_bids_recordings"]
