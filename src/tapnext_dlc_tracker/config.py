from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrackerConfig:
    keypoint_names: list[str]
    tapnext_window_seconds: float = 5.0
    dlc_supervision_stride_seconds: float = 3.0
    tapnext_confidence_floor: float = 0.45
    dlc_confidence_floor: float = 0.7
    dlc_force_replace: bool = False
    dlc_max_jump_pixels: float = 80.0
    max_query_age_seconds: float = 6.0
    fuse_tolerance_seconds: float = 0.1


def load_config(path: str | Path) -> TrackerConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    return TrackerConfig(
        keypoint_names=list(raw.get("keypoint_names", [])),
        tapnext_window_seconds=float(raw.get("tapnext_window_seconds", 5.0)),
        dlc_supervision_stride_seconds=float(raw.get("dlc_supervision_stride_seconds", 3.0)),
        tapnext_confidence_floor=float(raw.get("tapnext_confidence_floor", 0.45)),
        dlc_confidence_floor=float(raw.get("dlc_confidence_floor", 0.7)),
        dlc_force_replace=bool(raw.get("dlc_force_replace", False)),
        dlc_max_jump_pixels=float(raw.get("dlc_max_jump_pixels", 80.0)),
        max_query_age_seconds=float(raw.get("max_query_age_seconds", 6.0)),
        fuse_tolerance_seconds=float(raw.get("fuse_tolerance_seconds", 0.1)),
    )
