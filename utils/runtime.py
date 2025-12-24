"""Runtime helpers for qEEG executions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def ensure_output_tree(output_dir: Path) -> Dict[str, Path]:
    """Create the timestamped output tree for the current pipeline run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": output_dir,
        "results_csv": output_dir / "qEEG_result.csv",
        "segment_csv": output_dir / "qEEG_segment_result.csv",
        "microstate_csv": output_dir / "microstate_result.csv",
        "qc_html": output_dir / "QC.html",
        "microstate_qc_html": output_dir / "microstate_QC.html",
        "log_file": log_dir / "pipeline.log",
    }


def configure_logging(log_file: Path, level: str) -> None:
    """Send structured logging to both console and the run-specific log file."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


__all__ = ["configure_logging", "ensure_output_tree", "LOG_FORMAT"]
