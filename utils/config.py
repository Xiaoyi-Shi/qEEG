"""Configuration helpers for qEEG pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


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


__all__ = ["load_config", "resolve_path"]
