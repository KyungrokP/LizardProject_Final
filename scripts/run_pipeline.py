from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from tapnext_dlc_tracker.config import load_config
from tapnext_dlc_tracker.npz_queries import load_query_frames
from tapnext_dlc_tracker.pipeline import HybridTrackingPipeline
from tapnext_dlc_tracker.tapnext_client import (
    LocalTorchTapNextClient,
    MockTapNextClient,
    RemoteTapNextClient,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    video: str = typer.Option(..., help="Path to source video"),
    duration: float = typer.Option(..., min=0.01, help="Video duration in seconds"),
    config: str = typer.Option("configs/default.yaml", help="Tracker YAML config"),
    supervision_npz: str = typer.Option(
        "", help="NPZ file with DeepLabCut query frames (3-second cadence)"
    ),
    output_npz: str = typer.Option(
        "outputs/tracking_output.npz", help="Output merged tracking npz path"
    ),
    tapnext_backend: Literal["mock", "local_torch", "remote"] = typer.Option(
        "mock", help="TapNext backend"
    ),
    tapnext_ckpt: str = typer.Option(
        "/Users/kp/bootstapnext_ckpt.npz",
        help="Path to TAPNext .npz checkpoint for local_torch backend",
    ),
    tapnext_device: str = typer.Option("cpu", help="Torch device for local_torch backend"),
    tapnext_coord_order: Literal["xy", "yx"] = typer.Option(
        "yx",
        help="Coordinate order expected by TAPNext model (xy or yx)",
    ),
    tapnext_sketch_tokens: int = typer.Option(
        0,
        min=0,
        help="Experimental: sketch token count for ViT blocks (0 disables sketching)",
    ),
    tapnext_sketch_seed: int = typer.Option(
        0,
        help="Random seed for sketch matrix generation",
    ),
) -> None:
    cfg = load_config(config)
    sup_frames = []
    if supervision_npz:
        _, sup_frames = load_query_frames(supervision_npz)

    if tapnext_backend == "mock":
        tapnext = MockTapNextClient(
            max_window_seconds=cfg.tapnext_window_seconds,
            default_keypoints=cfg.keypoint_names,
        )
    elif tapnext_backend == "local_torch":
        sketch_tokens = tapnext_sketch_tokens if tapnext_sketch_tokens > 0 else None
        tapnext = LocalTorchTapNextClient(
            checkpoint_path=tapnext_ckpt,
            max_window_seconds=cfg.tapnext_window_seconds,
            device=tapnext_device,
            coord_order=tapnext_coord_order,
            sketch_tokens=sketch_tokens,
            sketch_seed=tapnext_sketch_seed,
        )
    else:
        tapnext = RemoteTapNextClient(max_window_seconds=cfg.tapnext_window_seconds)

    pipeline = HybridTrackingPipeline(tapnext=tapnext, config=cfg, supervision_frames=sup_frames)
    result = pipeline.run(video_path=video, duration_s=duration)

    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    pipeline.write_points_npz(out, result.points)

    typer.echo(f"Windows processed: {result.windows_run}")
    typer.echo(f"Supervision snapshots used: {result.supervision_hits}")
    typer.echo(f"Saved merged points: {out}")


if __name__ == "__main__":
    app()
