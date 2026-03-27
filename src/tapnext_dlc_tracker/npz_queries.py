from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import QueryFrame


def save_query_frames(path: str | Path, keypoint_names: list[str], frames: list[QueryFrame]) -> None:
    out = Path(path)
    if not frames:
        raise ValueError("frames is empty")
    if not keypoint_names:
        raise ValueError("keypoint_names is empty")

    timestamps = np.array([f.timestamp_s for f in frames], dtype=np.float32)
    coords = np.zeros((len(frames), len(keypoint_names), 2), dtype=np.float32)
    likelihoods = np.zeros((len(frames), len(keypoint_names)), dtype=np.float32)

    for i, frame in enumerate(frames):
        for j, name in enumerate(keypoint_names):
            xy = frame.points.get(name)
            if xy is None:
                coords[i, j] = np.array([np.nan, np.nan], dtype=np.float32)
                likelihoods[i, j] = 0.0
                continue
            coords[i, j] = np.array([xy[0], xy[1]], dtype=np.float32)
            likelihoods[i, j] = float(frame.likelihoods.get(name, 0.0))

    np.savez_compressed(
        out,
        timestamps=timestamps,
        keypoint_names=np.array(keypoint_names),
        coords=coords,
        likelihoods=likelihoods,
    )


def load_query_frames(path: str | Path) -> tuple[list[str], list[QueryFrame]]:
    data = np.load(Path(path), allow_pickle=False)
    keypoint_names = [str(v) for v in data["keypoint_names"].tolist()]
    timestamps = data["timestamps"]
    coords = data["coords"]
    likelihoods = data["likelihoods"]

    frames: list[QueryFrame] = []
    for i in range(len(timestamps)):
        points: dict[str, tuple[float, float]] = {}
        probs: dict[str, float] = {}
        for j, name in enumerate(keypoint_names):
            x, y = float(coords[i, j, 0]), float(coords[i, j, 1])
            conf = float(likelihoods[i, j])
            if np.isnan(x) or np.isnan(y):
                continue
            points[name] = (x, y)
            probs[name] = conf
        frames.append(
            QueryFrame(
                timestamp_s=float(timestamps[i]),
                points=points,
                likelihoods=probs,
            )
        )
    return keypoint_names, frames


def query_at_or_before(
    frames: list[QueryFrame],
    timestamp_s: float,
    max_age_s: float | None = None,
) -> QueryFrame | None:
    best: QueryFrame | None = None
    for frame in frames:
        if frame.timestamp_s <= timestamp_s:
            best = frame
        else:
            break
    if best is None:
        return None
    if max_age_s is not None and (timestamp_s - best.timestamp_s) > max_age_s:
        return None
    return best

