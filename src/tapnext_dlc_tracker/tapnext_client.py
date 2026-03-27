from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

from .tapnext_torch_runtime import load_tapnext_torch_model
from .types import TrackPoint


@dataclass(frozen=True)
class WindowRequest:
    video_path: str
    start_s: float
    end_s: float
    query_points: dict[str, tuple[float, float]] | None = None


class TapNextClient(Protocol):
    max_window_seconds: float

    def track_window(self, request: WindowRequest) -> list[TrackPoint]:
        ...


class RemoteTapNextClient:
    """Adapter shell for your real TapNext service call."""

    def __init__(self, max_window_seconds: float = 5.0) -> None:
        self.max_window_seconds = max_window_seconds

    def track_window(self, request: WindowRequest) -> list[TrackPoint]:
        if request.end_s - request.start_s > self.max_window_seconds + 1e-9:
            raise ValueError("TapNext request exceeds max window size")
        raise NotImplementedError(
            "Replace RemoteTapNextClient.track_window with your TapNext API integration."
        )


class LocalTorchTapNextClient:
    """Runs TAPNext locally using the torch demo model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        max_window_seconds: float = 5.0,
        image_size: tuple[int, int] = (256, 256),
        device: str = "cpu",
        coord_order: str = "yx",
        sketch_tokens: int | None = None,
        sketch_seed: int = 0,
    ) -> None:
        if coord_order not in {"xy", "yx"}:
            raise ValueError("coord_order must be one of: xy, yx")
        self.max_window_seconds = max_window_seconds
        self.checkpoint_path = str(checkpoint_path)
        self.image_size = image_size
        self.device = device
        self.coord_order = coord_order
        self.sketch_tokens = sketch_tokens
        self.sketch_seed = sketch_seed
        self._model: torch.nn.Module | None = None

    def track_window(self, request: WindowRequest) -> list[TrackPoint]:
        if request.end_s - request.start_s > self.max_window_seconds + 1e-9:
            raise ValueError("TapNext request exceeds max window size")
        if not request.query_points:
            raise ValueError(
                "TapNext local torch backend requires query_points for each window"
            )

        model = self._ensure_model()
        frames, frame_numbers, fps, orig_w, orig_h = self._load_window_frames(
            request.video_path, request.start_s, request.end_s
        )
        if not frames:
            return []

        h, w = self.image_size
        sx = w / orig_w
        sy = h / orig_h
        query_items = list(request.query_points.items())
        query_names = [k for k, _ in query_items]

        query = np.zeros((1, len(query_items), 3), dtype=np.float32)
        for i, (_, (x, y)) in enumerate(query_items):
            query[0, i, 0] = 0.0
            if self.coord_order == "xy":
                query[0, i, 1] = x * sx
                query[0, i, 2] = y * sy
            else:
                query[0, i, 1] = y * sy
                query[0, i, 2] = x * sx

        video = torch.from_numpy(np.asarray(frames, dtype=np.float32)).to(self.device)
        video = video.unsqueeze(0)
        query_t = torch.from_numpy(query).to(self.device)

        with torch.no_grad():
            pred_tracks, _, visible_logits, state = model(
                video=video[:, :1], query_points=query_t
            )
            tracks_parts = [pred_tracks]
            visible_parts = [visible_logits]
            for i in range(1, video.shape[1]):
                curr_tracks, _, curr_visible, state = model(
                    video=video[:, i : i + 1], state=state
                )
                tracks_parts.append(curr_tracks)
                visible_parts.append(curr_visible)

        tracks = torch.cat(tracks_parts, dim=1)[0].cpu().numpy()  # [T, Q, 2]
        visible = torch.sigmoid(torch.cat(visible_parts, dim=1)[0, :, :, 0]).cpu().numpy()

        points: list[TrackPoint] = []
        for t_idx, frame_no in enumerate(frame_numbers):
            ts = round(float(frame_no / fps), 6)
            for q_idx, name in enumerate(query_names):
                a = float(tracks[t_idx, q_idx, 0])
                b = float(tracks[t_idx, q_idx, 1])
                if self.coord_order == "xy":
                    x = a / sx
                    y = b / sy
                else:
                    y = a / sy
                    x = b / sx
                points.append(
                    TrackPoint(
                        keypoint=name,
                        timestamp_s=ts,
                        x=float(x),
                        y=float(y),
                        likelihood=float(visible[t_idx, q_idx]),
                        source="tapnext",
                    )
                )
        return points

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_tapnext_torch_model(
                checkpoint_path=self.checkpoint_path,
                image_size=self.image_size,
                device=self.device,
                sketch_tokens=self.sketch_tokens,
                sketch_seed=self.sketch_seed,
            )
        return self._model

    def _load_window_frames(
        self,
        video_path: str,
        start_s: float,
        end_s: float,
    ) -> tuple[list[np.ndarray], list[int], float, int, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        start_frame = max(0, int(np.floor(start_s * fps)))
        end_frame = max(start_frame + 1, int(np.ceil(end_s * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        h, w = self.image_size
        frames: list[np.ndarray] = []
        frame_numbers: list[int] = []
        orig_w = 0
        orig_h = 0

        cur = start_frame
        while cur < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if orig_w == 0:
                orig_h, orig_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(resized.astype(np.float32) / 255.0)
            frame_numbers.append(cur)
            cur += 1

        cap.release()
        return frames, frame_numbers, fps, orig_w, orig_h


class MockTapNextClient:
    """Deterministic mock for local pipeline testing."""

    def __init__(
        self,
        max_window_seconds: float = 5.0,
        fps: float = 30.0,
        default_keypoints: list[str] | None = None,
    ) -> None:
        self.max_window_seconds = max_window_seconds
        self.fps = fps
        self.default_keypoints = list(default_keypoints or [])

    def track_window(self, request: WindowRequest) -> list[TrackPoint]:
        if request.end_s - request.start_s > self.max_window_seconds + 1e-9:
            raise ValueError("TapNext request exceeds max window size")

        keypoints = sorted((request.query_points or {}).keys())
        if not keypoints:
            keypoints = self.default_keypoints or [
                "nose",
                "left_ear",
                "right_ear",
                "tail_base",
            ]

        num_steps = max(1, int(math.ceil((request.end_s - request.start_s) * self.fps)))
        points: list[TrackPoint] = []
        for i in range(num_steps + 1):
            t = request.start_s + ((request.end_s - request.start_s) * i / num_steps)
            for idx, name in enumerate(keypoints):
                base_x = 80 + 15 * idx
                base_y = 120 + 10 * idx
                qx, qy = (request.query_points or {}).get(name, (base_x, base_y))
                x = qx + math.sin(t * 0.8 + idx) * 2.5
                y = qy + math.cos(t * 0.6 + idx) * 2.5
                likelihood = 0.6 + 0.35 * abs(math.sin(t + idx * 0.1))
                points.append(
                    TrackPoint(
                        keypoint=name,
                        timestamp_s=round(t, 6),
                        x=float(x),
                        y=float(y),
                        likelihood=min(0.99, likelihood),
                        source="tapnext",
                    )
                )
        return points
