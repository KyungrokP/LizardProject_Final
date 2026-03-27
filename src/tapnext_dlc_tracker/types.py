from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPoint:
    keypoint: str
    timestamp_s: float
    x: float
    y: float
    likelihood: float
    source: str


@dataclass(frozen=True)
class QueryFrame:
    timestamp_s: float
    points: dict[str, tuple[float, float]]
    likelihoods: dict[str, float]

