"""Microstate feature helpers backed by pycrostates."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
from pycrostates.io import read_cluster

from utils.config import resolve_path

MICROSTATE_COLUMNS = ["subject_id", "kind", "state", "from_state", "to_state", "metric", "value"]
EMPTY_MICROSTATE_FRAME = pd.DataFrame(columns=MICROSTATE_COLUMNS)


def _normalize_enable_flag(value: Any, default: bool = False) -> bool:
    """Coerce mixed representations of boolean flags into strict booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "1", "on", "enabled"}:
            return True
        if normalized in {"no", "false", "0", "off", "disabled"}:
            return False
    raise ValueError(f"Cannot interpret enable flag value: {value!r}")


def _ensure_path(value: str | Path | None, base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    return resolve_path(str(value), base_dir)


def _ensure_str_list(values: Iterable[Any] | None) -> Optional[List[str]]:
    if values is None:
        return None
    result: List[str] = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        else:
            result.append(str(value))
    return result or None


@dataclass(frozen=True)
class MicrostateConfig:
    """Parsed microstate configuration."""

    enable: bool = False
    template_path: Optional[Path] = None
    picks: Optional[List[str]] = None
    factor: int = 0
    half_window_size: int = 1
    tol: float = 1e-5
    min_segment_length: int = 0
    reject_edges: bool = True
    reject_by_annotation: bool = True
    norm_gfp: bool = True
    return_dist: bool = False
    entropy_ignore_repetitions: bool = False
    entropy_log_base: float | str = 2
    transition_stat: str = "probability"
    transition_ignore_repetitions: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None, *, base_dir: Path) -> "MicrostateConfig":
        if not params:
            return cls()

        enable = _normalize_enable_flag(params.get("enable"), default=False)
        template_path = _ensure_path(params.get("template_path"), base_dir)
        if enable and template_path is None:
            raise ValueError("microstate.template_path is required when microstate is enabled.")
        if enable and template_path is not None and not template_path.exists():
            raise FileNotFoundError(f"Microstate template does not exist: {template_path}")

        picks = _ensure_str_list(params.get("picks"))
        factor = int(params.get("factor", cls.factor))
        half_window_size = int(params.get("half_window_size", cls.half_window_size))
        min_segment_length = int(params.get("min_segment_length", cls.min_segment_length))
        tol = float(params.get("tol", cls.tol))
        if factor < 0:
            raise ValueError("microstate.factor must be non-negative.")
        if half_window_size < 1:
            raise ValueError("microstate.half_window_size must be >= 1.")
        if min_segment_length < 0:
            raise ValueError("microstate.min_segment_length must be >= 0.")
        if tol <= 0:
            raise ValueError("microstate.tol must be positive.")
        reject_edges = _normalize_enable_flag(params.get("reject_edges"), default=cls.reject_edges)
        reject_by_annotation = _normalize_enable_flag(
            params.get("reject_by_annotation"), default=cls.reject_by_annotation
        )
        norm_gfp = _normalize_enable_flag(params.get("norm_gfp"), default=cls.norm_gfp)
        return_dist = _normalize_enable_flag(params.get("return_dist"), default=cls.return_dist)
        entropy_ignore_repetitions = _normalize_enable_flag(
            params.get("entropy_ignore_repetitions"), default=cls.entropy_ignore_repetitions
        )
        entropy_log_base = params.get("entropy_log_base", cls.entropy_log_base)
        transition_stat = str(params.get("transition_stat", cls.transition_stat))
        valid_stats = {"probability", "count", "proportion", "percent"}
        if transition_stat not in valid_stats:
            raise ValueError(f"transition_stat must be one of {sorted(valid_stats)}.")
        transition_ignore_repetitions = _normalize_enable_flag(
            params.get("transition_ignore_repetitions"), default=cls.transition_ignore_repetitions
        )

        return cls(
            enable=enable,
            template_path=template_path,
            picks=picks,
            factor=factor,
            half_window_size=half_window_size,
            tol=tol,
            min_segment_length=min_segment_length,
            reject_edges=reject_edges,
            reject_by_annotation=reject_by_annotation,
            norm_gfp=norm_gfp,
            return_dist=return_dist,
            entropy_ignore_repetitions=entropy_ignore_repetitions,
            entropy_log_base=entropy_log_base,
            transition_stat=transition_stat,
            transition_ignore_repetitions=transition_ignore_repetitions,
        )


def _tidy_parameters(subject_id: str, params: Dict[str, Any], *, logger: logging.Logger) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            # Skip distribution arrays; retaining them would bloat the tidy output.
            logger.debug("Skipping distribution array in microstate parameters: %s (shape=%s)", key, value.shape)
            continue
        if isinstance(value, (list, tuple, dict)):
            logger.debug("Skipping non-scalar parameter '%s' in microstate parameters.", key)
            continue
        if key == "unlabeled":
            rows.append(
                {
                    "subject_id": subject_id,
                    "kind": "parameters",
                    "state": None,
                    "from_state": None,
                    "to_state": None,
                    "metric": "unlabeled",
                    "value": float(value),
                }
            )
            continue
        if "_" in key:
            state, metric = key.split("_", 1)
        else:
            state, metric = None, key
        rows.append(
            {
                "subject_id": subject_id,
                "kind": "parameters",
                "state": state,
                "from_state": None,
                "to_state": None,
                "metric": metric,
                "value": float(value),
            }
        )
    return pd.DataFrame(rows, columns=MICROSTATE_COLUMNS)


def _tidy_entropy(subject_id: str, entropy_value: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "kind": "entropy",
                "state": None,
                "from_state": None,
                "to_state": None,
                "metric": "entropy",
                "value": float(entropy_value),
            }
        ],
        columns=MICROSTATE_COLUMNS,
    )


def _tidy_transition_matrix(
    subject_id: str,
    matrix: np.ndarray,
    state_names: List[str],
    *,
    stat: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, from_state in enumerate(state_names):
        for j, to_state in enumerate(state_names):
            rows.append(
                {
                    "subject_id": subject_id,
                    "kind": "transition",
                    "state": None,
                    "from_state": from_state,
                    "to_state": to_state,
                    "metric": stat,
                    "value": float(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows, columns=MICROSTATE_COLUMNS)


def compute_microstate_metrics(
    raw: "mne.io.BaseRaw",
    subject_id: str,
    config: MicrostateConfig,
    *,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute microstate parameters/entropy/transitions for a preprocessed Raw."""
    if not config.enable:
        return EMPTY_MICROSTATE_FRAME.copy()
    if config.template_path is None:
        logger.error("Microstate enabled but no template_path provided.")
        return EMPTY_MICROSTATE_FRAME.copy()

    logger.debug("Loading microstate template from %s", config.template_path)
    cluster = read_cluster(str(config.template_path))
    logger.debug(
        "Predicting microstates (half_window_size=%d, min_segment_length=%d, reject_by_annotation=%s)",
        config.half_window_size,
        config.min_segment_length,
        config.reject_by_annotation,
    )
    segmentation = cluster.predict(
        raw,
        picks=config.picks,
        factor=config.factor,
        half_window_size=config.half_window_size,
        tol=config.tol,
        min_segment_length=config.min_segment_length,
        reject_edges=config.reject_edges,
        reject_by_annotation=config.reject_by_annotation,
    )

    frames: List[pd.DataFrame] = []

    params = segmentation.compute_parameters(norm_gfp=config.norm_gfp, return_dist=config.return_dist)
    frames.append(_tidy_parameters(subject_id, params, logger=logger))

    entropy_value = segmentation.entropy(
        ignore_repetitions=config.entropy_ignore_repetitions,
        log_base=config.entropy_log_base,
    )
    frames.append(_tidy_entropy(subject_id, entropy_value))

    transition_matrix = segmentation.compute_transition_matrix(
        stat=config.transition_stat,
        ignore_repetitions=config.transition_ignore_repetitions,
    )
    frames.append(
        _tidy_transition_matrix(
            subject_id,
            transition_matrix,
            state_names=segmentation.cluster_names,
            stat=config.transition_stat,
        )
    )

    return pd.concat(frames, ignore_index=True) if frames else EMPTY_MICROSTATE_FRAME.copy()
__all__ = [
    "EMPTY_MICROSTATE_FRAME",
    "MICROSTATE_COLUMNS",
    "MicrostateConfig",
    "compute_microstate_metrics",
    "_normalize_enable_flag",
]
