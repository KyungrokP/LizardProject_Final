from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from tapnext_dlc_tracker.config import load_config
from tapnext_dlc_tracker.deeplabcut_bridge import (
    analyze_video_with_deeplabcut,
    build_sampled_video,
    csv_to_query_frames,
)
from tapnext_dlc_tracker.npz_queries import save_query_frames

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dlc_config: str = typer.Option(..., help="Path to DeepLabCut config.yaml"),
    video: str = typer.Option(..., help="Video path for DeepLabCut analysis"),
    fps: float = typer.Option(
        0.0,
        min=0.0,
        help="Source video fps (required for --mode full_video and raw CSV mode)",
    ),
    tracker_config: str = typer.Option("configs/default.yaml", help="Tracker config"),
    mode: Literal["sampled_3s", "full_video"] = typer.Option(
        "sampled_3s",
        help="sampled_3s: run DLC only on sampled frames at tracker stride; full_video: analyze every frame",
    ),
    out_npz: str = typer.Option(
        "outputs/dlc_queries_3s.npz",
        help="Output npz containing sampled query frames",
    ),
    analysis_dir: str = typer.Option(
        "outputs/dlc_analysis", help="Directory for raw DeepLabCut outputs"
    ),
    csv_path: str = typer.Option(
        "",
        help="Skip analyze_videos and use existing DeepLabCut CSV path directly",
    ),
    sampled_video_path: str = typer.Option(
        "outputs/dlc_analysis/dlc_sampled_stride_video.mp4",
        help="Temporary sampled video path used when --mode sampled_3s",
    ),
) -> None:
    cfg = load_config(tracker_config)
    effective_fps = fps
    if csv_path:
        csv_file = Path(csv_path)
        if effective_fps <= 0:
            effective_fps = 1.0 / cfg.dlc_supervision_stride_seconds
    elif mode == "sampled_3s":
        sampled_fps, written = build_sampled_video(
            video_path=video,
            stride_seconds=cfg.dlc_supervision_stride_seconds,
            output_video_path=sampled_video_path,
        )
        effective_fps = sampled_fps
        typer.echo(
            f"Built sampled video at {cfg.dlc_supervision_stride_seconds}s stride "
            f"({written} frames): {sampled_video_path}"
        )
        csv_file = analyze_video_with_deeplabcut(
            dlc_config_path=dlc_config,
            video_path=sampled_video_path,
            output_dir=analysis_dir,
        )
    else:
        if effective_fps <= 0:
            raise typer.BadParameter("--fps must be provided when --mode full_video")
        csv_file = analyze_video_with_deeplabcut(
            dlc_config_path=dlc_config,
            video_path=video,
            output_dir=analysis_dir,
        )

    frames = csv_to_query_frames(
        csv_path=csv_file,
        keypoint_names=cfg.keypoint_names,
        fps=effective_fps,
        stride_seconds=cfg.dlc_supervision_stride_seconds,
        confidence_floor=cfg.dlc_confidence_floor,
    )
    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_query_frames(out, cfg.keypoint_names, frames)
    typer.echo(f"Saved {len(frames)} supervision frames: {out}")


if __name__ == "__main__":
    app()
