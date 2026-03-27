from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import TrackerConfig
from .fusion import apply_supervision
from .npz_queries import query_at_or_before
from .tapnext_client import TapNextClient, WindowRequest
from .timegrid import make_supervision_times, make_windows
from .types import QueryFrame, TrackPoint


@dataclass(frozen=True)
class PipelineResult:
    points: list[TrackPoint]
    supervision_hits: int
    windows_run: int


class HybridTrackingPipeline:
    def __init__(
        self,
        tapnext: TapNextClient,
        config: TrackerConfig,
        supervision_frames: list[QueryFrame] | None = None,
    ) -> None:
        self.tapnext = tapnext
        self.config = config
        self.supervision_frames = sorted(
            supervision_frames or [], key=lambda f: f.timestamp_s
        )

    def run(self, video_path: str, duration_s: float) -> PipelineResult:
        windows = make_windows(0.0, duration_s, self.config.tapnext_window_seconds)
        _sup_grid = make_supervision_times(
            0.0, duration_s, self.config.dlc_supervision_stride_seconds
        )
        all_points: list[TrackPoint] = []
        query_points: dict[str, tuple[float, float]] | None = None
        supervision_hits = 0

        for start_s, end_s in windows:
            if query_points is None:
                seed = query_at_or_before(
                    self.supervision_frames,
                    timestamp_s=start_s,
                    max_age_s=self.config.max_query_age_seconds,
                )
                if seed is not None and seed.points:
                    query_points = dict(seed.points)

            request = WindowRequest(
                video_path=video_path,
                start_s=start_s,
                end_s=end_s,
                query_points=query_points,
            )
            window_points = self.tapnext.track_window(request)
            # Apply every supervision snapshot that falls in this window.
            in_window = self._supervisions_in_window(start_s=start_s, end_s=end_s)
            if in_window:
                supervision_hits += len(in_window)
                for supervision in in_window:
                    window_points = apply_supervision(
                        points=window_points,
                        supervision=supervision,
                        tapnext_conf_floor=self.config.tapnext_confidence_floor,
                        dlc_conf_floor=self.config.dlc_confidence_floor,
                        tolerance_s=self.config.fuse_tolerance_seconds,
                        dlc_force_replace=self.config.dlc_force_replace,
                        dlc_max_jump_pixels=self.config.dlc_max_jump_pixels,
                    )
                last_sup_t = in_window[-1].timestamp_s
                query_points = self._points_near_timestamp(
                    points=window_points,
                    timestamp_s=last_sup_t,
                    tolerance_s=self.config.fuse_tolerance_seconds,
                )
                if not query_points:
                    query_points = self._latest_points(window_points)
            else:
                # Fallback for sparse supervision: try the latest older snapshot.
                supervision = query_at_or_before(
                    self.supervision_frames,
                    timestamp_s=start_s,
                    max_age_s=self.config.max_query_age_seconds,
                )
                if supervision is not None:
                    supervision_hits += 1
                    window_points = apply_supervision(
                        points=window_points,
                        supervision=supervision,
                        tapnext_conf_floor=self.config.tapnext_confidence_floor,
                        dlc_conf_floor=self.config.dlc_confidence_floor,
                        tolerance_s=self.config.fuse_tolerance_seconds,
                        dlc_force_replace=self.config.dlc_force_replace,
                        dlc_max_jump_pixels=self.config.dlc_max_jump_pixels,
                    )
                    query_points = self._points_near_timestamp(
                        points=window_points,
                        timestamp_s=supervision.timestamp_s,
                        tolerance_s=self.config.fuse_tolerance_seconds,
                    )
                    if not query_points:
                        query_points = self._latest_points(window_points)
                else:
                    query_points = self._latest_points(window_points)

            all_points.extend(window_points)

        return PipelineResult(
            points=all_points,
            supervision_hits=supervision_hits,
            windows_run=len(windows),
        )

    def _supervisions_in_window(self, start_s: float, end_s: float) -> list[QueryFrame]:
        tol = self.config.fuse_tolerance_seconds
        lo = start_s - tol
        hi = end_s + tol
        return [f for f in self.supervision_frames if lo <= f.timestamp_s <= hi]

    @staticmethod
    def _latest_points(points: list[TrackPoint]) -> dict[str, tuple[float, float]]:
        latest_t = max((p.timestamp_s for p in points), default=0.0)
        out: dict[str, tuple[float, float]] = {}
        for p in points:
            if abs(p.timestamp_s - latest_t) < 1e-9:
                out[p.keypoint] = (p.x, p.y)
        return out

    @staticmethod
    def _points_near_timestamp(
        points: list[TrackPoint],
        timestamp_s: float,
        tolerance_s: float,
    ) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for p in points:
            if abs(p.timestamp_s - timestamp_s) <= tolerance_s:
                out[p.keypoint] = (p.x, p.y)
        return out

    @staticmethod
    def write_points_npz(path: str | Path, points: list[TrackPoint]) -> None:
        out = Path(path)
        keypoints = sorted({p.keypoint for p in points})
        times = sorted({p.timestamp_s for p in points})
        lookup = {(p.timestamp_s, p.keypoint): p for p in points}

        coords = np.full((len(times), len(keypoints), 2), np.nan, dtype=np.float32)
        likelihoods = np.zeros((len(times), len(keypoints)), dtype=np.float32)
        sources = np.full((len(times), len(keypoints)), "", dtype="<U16")

        for i, t in enumerate(times):
            for j, name in enumerate(keypoints):
                p = lookup.get((t, name))
                if p is None:
                    continue
                coords[i, j, 0] = p.x
                coords[i, j, 1] = p.y
                likelihoods[i, j] = p.likelihood
                sources[i, j] = p.source

        np.savez_compressed(
            out,
            timestamps=np.array(times, dtype=np.float32),
            keypoint_names=np.array(keypoints),
            coords=coords,
            likelihoods=likelihoods,
            sources=sources,
        )
