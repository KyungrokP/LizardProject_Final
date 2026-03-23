from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import typer

app = typer.Typer(add_completion=False)


def _pick_row(timestamps: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(timestamps, t, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    left = idx - 1
    if abs(float(timestamps[idx]) - t) < abs(float(timestamps[left]) - t):
        return idx
    return left


@app.command()
def main(
    video: str = typer.Option(..., help="Input video path"),
    tracking_npz: str = typer.Option(..., help="Tracking npz path"),
    output_video: str = typer.Option(
        "outputs/tracking_overlay.mp4", help="Output overlay video"
    ),
    color_by_source: bool = typer.Option(
        False,
        help="If true, color points by source (deeplabcut=red, tapnext=blue, other=green)",
    ),
    radius: int = typer.Option(6, min=1, help="Circle radius"),
    font_scale: float = typer.Option(0.45, min=0.1, help="Label font scale"),
) -> None:
    src = Path(video)
    npz = Path(tracking_npz)
    out = Path(output_video)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz, allow_pickle=False)
    timestamps = data["timestamps"].astype(np.float64)
    keypoints = [str(v) for v in data["keypoint_names"].tolist()]
    coords = data["coords"].astype(np.float64)
    sources = data["sources"] if "sources" in data.files else None

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {out}")

    palette = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 220, 80),
        (255, 80, 255),
        (80, 255, 255),
    ]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        row = _pick_row(timestamps, t)
        for j, name in enumerate(keypoints):
            x, y = float(coords[row, j, 0]), float(coords[row, j, 1])
            if np.isnan(x) or np.isnan(y):
                continue
            xi = int(round(x))
            yi = int(round(y))
            color = palette[j % len(palette)]
            source_label = ""
            if color_by_source and sources is not None:
                source_label = str(sources[row, j])
                if source_label == "deeplabcut":
                    color = (0, 0, 255)
                elif source_label == "tapnext":
                    color = (255, 80, 80)
                else:
                    color = (80, 255, 80)
            cv2.circle(frame, (xi, yi), radius, color, -1)
            label = name
            if color_by_source and source_label:
                label = f"{name}:{source_label[:3]}"
            cv2.putText(
                frame,
                label,
                (xi + 8, yi - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    typer.echo(f"Saved overlay video: {out}")


if __name__ == "__main__":
    app()
