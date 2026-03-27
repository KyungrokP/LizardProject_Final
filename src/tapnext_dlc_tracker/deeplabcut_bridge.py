from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from .types import QueryFrame


def analyze_video_with_deeplabcut(
    dlc_config_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
) -> Path:
    try:
        import deeplabcut  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DeepLabCut is not installed. Install optional deps: pip install .[dlc]"
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    deeplabcut.analyze_videos(
        str(dlc_config_path),
        [str(video_path)],
        destfolder=str(output),
        save_as_csv=True,
    )

    csv_files = sorted(output.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError("No DeepLabCut CSV output found")
    return csv_files[0]


def build_sampled_video(
    video_path: str | Path,
    stride_seconds: float,
    output_video_path: str | Path,
) -> tuple[float, int]:
    """Builds a sampled video that keeps one frame every `stride_seconds`."""
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        cap.release()
        raise RuntimeError("Could not read source video FPS")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step_frames = max(1, int(round(stride_seconds * src_fps)))
    sampled_fps = 1.0 / stride_seconds

    out_path = Path(output_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    written = 0

    try:
        for frame_idx in range(0, max(total_frames, 1), step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    sampled_fps,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output writer: {out_path}")

            writer.write(frame)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise RuntimeError("No frames were written to sampled video")
    return sampled_fps, written


def csv_to_query_frames(
    csv_path: str | Path,
    keypoint_names: list[str],
    fps: float,
    stride_seconds: float,
    confidence_floor: float,
) -> list[QueryFrame]:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")

    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    stride_frames = max(1, int(round(stride_seconds * fps)))

    frames: list[QueryFrame] = []
    for i in range(0, len(df), stride_frames):
        row = df.iloc[i]
        points: dict[str, tuple[float, float]] = {}
        likelihoods: dict[str, float] = {}
        for name in keypoint_names:
            x = _get_col_value(row, name, "x")
            y = _get_col_value(row, name, "y")
            p = _get_col_value(row, name, "likelihood")
            if x is None or y is None or p is None:
                continue
            conf = float(p)
            if conf < confidence_floor:
                continue
            points[name] = (float(x), float(y))
            likelihoods[name] = conf

        frames.append(
            QueryFrame(
                timestamp_s=float(i / fps),
                points=points,
                likelihoods=likelihoods,
            )
        )
    return frames


def _get_col_value(row: pd.Series, bodypart: str, coord: str) -> float | None:
    for col in row.index:
        if len(col) != 3:
            continue
        _, part, axis = col
        if part == bodypart and axis == coord:
            return float(row[col])
    return None
