from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import cv2
import numpy as np
import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tapnext_dlc_tracker.config import load_config
from tapnext_dlc_tracker.deeplabcut_bridge import csv_to_query_frames
from tapnext_dlc_tracker.npz_queries import load_query_frames, save_query_frames
from tapnext_dlc_tracker.pipeline import HybridTrackingPipeline
from tapnext_dlc_tracker.tapnext_client import LocalTorchTapNextClient

app = typer.Typer(add_completion=False)

LOGGER = logging.getLogger("slurm_chunked_dlc_tapnext")


@dataclass(frozen=True)
class ChunkJob:
    chunk_id: int
    chunk_path: str
    start_frame: int
    end_frame: int
    fps: float


@dataclass(frozen=True)
class DlcChunkResult:
    chunk_id: int
    chunk_path: str
    start_frame: int
    end_frame: int
    fps: float
    dlc_csv_path: str
    anchor_npz_path: str
    error: str = ""


@dataclass(frozen=True)
class TapChunkResult:
    chunk_id: int
    start_frame: int
    end_frame: int
    fps: float
    track_npz_path: str
    supervision_hits: int
    windows_run: int
    error: str = ""


@dataclass(frozen=True)
class DlcWorkerConfig:
    dlc_config_path: str
    tracker_config_path: str
    output_root: str
    dlc_device: str
    dlc_shuffle: int
    dlc_batch_size: int
    dlc_anchor_stride_seconds: float
    dlc_confidence_floor_override: float
    dlc_disable_multithreading: bool


@dataclass(frozen=True)
class TapWorkerConfig:
    tracker_config_path: str
    tapnext_ckpt: str
    tapnext_device: str
    tapnext_coord_order: str
    tapnext_sketch_tokens: int
    tapnext_sketch_seed: int
    output_root: str


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
    )


def _video_metadata(video_path: str | Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS from: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Could not read frame count from: {video_path}")
    return fps, frame_count, width, height


def _write_chunk(
    video_path: str | Path,
    out_path: str | Path,
    fps: float,
    width: int,
    height: int,
    start_frame: int,
    end_frame: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {out}")

    cur = start_frame
    try:
        while cur < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            cur += 1
    finally:
        writer.release()
        cap.release()
    return cur


def split_video_into_chunks(
    video_path: str | Path,
    chunk_seconds: float,
    overlap_seconds: float,
    chunks_dir: str | Path,
) -> tuple[float, int, list[ChunkJob]]:
    fps, frame_count, width, height = _video_metadata(video_path)
    chunk_frames = max(1, int(round(chunk_seconds * fps)))
    overlap_frames = max(0, int(round(overlap_seconds * fps)))
    step = chunk_frames - overlap_frames
    if step <= 0:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    jobs: list[ChunkJob] = []
    start = 0
    chunk_id = 0
    while start < frame_count:
        end = min(frame_count, start + chunk_frames)
        chunk_path = Path(chunks_dir) / f"chunk_{chunk_id:05d}.mp4"
        actual_end = _write_chunk(
            video_path=video_path,
            out_path=chunk_path,
            fps=fps,
            width=width,
            height=height,
            start_frame=start,
            end_frame=end,
        )
        if actual_end <= start:
            break
        jobs.append(
            ChunkJob(
                chunk_id=chunk_id,
                chunk_path=str(chunk_path),
                start_frame=start,
                end_frame=actual_end,
                fps=fps,
            )
        )
        if actual_end >= frame_count:
            break
        start += step
        chunk_id += 1

    if not jobs:
        raise RuntimeError("No chunks were generated")
    return fps, frame_count, jobs


def _find_latest_csv(output_dir: str | Path) -> Path:
    files = sorted(Path(output_dir).glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No DLC CSV found in {output_dir}")
    return files[0]


def _ensure_seed_anchor(frames: list[Any]) -> list[Any]:
    if not frames:
        return frames
    if frames[0].points:
        return frames
    first_with_points = None
    for fr in frames:
        if fr.points:
            first_with_points = fr
            break
    if first_with_points is None:
        return frames
    seeded = [type(first_with_points)(timestamp_s=0.0, points=first_with_points.points, likelihoods=first_with_points.likelihoods)]
    seeded.extend(frames)
    return seeded


def dlc_worker(
    queue_in: mp.Queue[ChunkJob | None],
    queue_out: mp.Queue[DlcChunkResult | None],
    worker_cfg: DlcWorkerConfig,
) -> None:
    _setup_logging(verbose=False)
    try:
        import deeplabcut  # type: ignore
    except Exception as exc:
        queue_out.put(
            DlcChunkResult(
                chunk_id=-1,
                chunk_path="",
                start_frame=0,
                end_frame=0,
                fps=0.0,
                dlc_csv_path="",
                anchor_npz_path="",
                error=f"DLC import failed: {exc}",
            )
        )
        queue_out.put(None)
        return

    tracker_cfg = load_config(worker_cfg.tracker_config_path)
    confidence_floor = (
        worker_cfg.dlc_confidence_floor_override
        if worker_cfg.dlc_confidence_floor_override >= 0
        else tracker_cfg.dlc_confidence_floor
    )

    while True:
        job = queue_in.get()
        if job is None:
            break
        try:
            LOGGER.info("DLC worker processing chunk %s (%s)", job.chunk_id, job.chunk_path)
            chunk_out = Path(worker_cfg.output_root) / "dlc_chunks" / f"chunk_{job.chunk_id:05d}"
            chunk_out.mkdir(parents=True, exist_ok=True)

            kwargs: dict[str, Any] = {}
            if worker_cfg.dlc_disable_multithreading:
                kwargs["inference_cfg"] = {
                    "multithreading": {"enabled": False, "queue_length": 1, "timeout": 30.0}
                }

            deeplabcut.analyze_videos(
                worker_cfg.dlc_config_path,
                [job.chunk_path],
                shuffle=worker_cfg.dlc_shuffle,
                destfolder=str(chunk_out),
                save_as_csv=True,
                batchsize=worker_cfg.dlc_batch_size,
                device=worker_cfg.dlc_device,
                **kwargs,
            )

            csv_path = _find_latest_csv(chunk_out)
            stride_s = worker_cfg.dlc_anchor_stride_seconds
            if stride_s <= 0:
                stride_s = 1.0 / job.fps

            frames = csv_to_query_frames(
                csv_path=csv_path,
                keypoint_names=tracker_cfg.keypoint_names,
                fps=job.fps,
                stride_seconds=stride_s,
                confidence_floor=confidence_floor,
            )
            frames = _ensure_seed_anchor(frames)
            if not frames:
                raise RuntimeError("No anchors were produced from DLC output")

            anchor_npz = chunk_out / f"chunk_{job.chunk_id:05d}_anchors.npz"
            save_query_frames(anchor_npz, tracker_cfg.keypoint_names, frames)

            queue_out.put(
                DlcChunkResult(
                    chunk_id=job.chunk_id,
                    chunk_path=job.chunk_path,
                    start_frame=job.start_frame,
                    end_frame=job.end_frame,
                    fps=job.fps,
                    dlc_csv_path=str(csv_path),
                    anchor_npz_path=str(anchor_npz),
                )
            )
        except Exception:
            queue_out.put(
                DlcChunkResult(
                    chunk_id=job.chunk_id,
                    chunk_path=job.chunk_path,
                    start_frame=job.start_frame,
                    end_frame=job.end_frame,
                    fps=job.fps,
                    dlc_csv_path="",
                    anchor_npz_path="",
                    error=traceback.format_exc(),
                )
            )
    queue_out.put(None)


def tapnext_worker(
    queue_in: mp.Queue[DlcChunkResult | None],
    queue_out: mp.Queue[TapChunkResult | None],
    worker_cfg: TapWorkerConfig,
) -> None:
    _setup_logging(verbose=False)
    tracker_cfg = load_config(worker_cfg.tracker_config_path)
    sketch_tokens = worker_cfg.tapnext_sketch_tokens if worker_cfg.tapnext_sketch_tokens > 0 else None
    tapnext = LocalTorchTapNextClient(
        checkpoint_path=worker_cfg.tapnext_ckpt,
        max_window_seconds=tracker_cfg.tapnext_window_seconds,
        device=worker_cfg.tapnext_device,
        coord_order=worker_cfg.tapnext_coord_order,
        sketch_tokens=sketch_tokens,
        sketch_seed=worker_cfg.tapnext_sketch_seed,
    )

    while True:
        dlc_res = queue_in.get()
        if dlc_res is None:
            break
        if dlc_res.error:
            queue_out.put(
                TapChunkResult(
                    chunk_id=dlc_res.chunk_id,
                    start_frame=dlc_res.start_frame,
                    end_frame=dlc_res.end_frame,
                    fps=dlc_res.fps,
                    track_npz_path="",
                    supervision_hits=0,
                    windows_run=0,
                    error=f"DLC worker error for chunk {dlc_res.chunk_id}:\n{dlc_res.error}",
                )
            )
            continue

        try:
            _, sup_frames = load_query_frames(dlc_res.anchor_npz_path)
            duration_s = (dlc_res.end_frame - dlc_res.start_frame) / dlc_res.fps
            pipeline = HybridTrackingPipeline(
                tapnext=tapnext,
                config=tracker_cfg,
                supervision_frames=sup_frames,
            )
            pipe_res = pipeline.run(
                video_path=dlc_res.chunk_path,
                duration_s=duration_s,
            )

            out_dir = Path(worker_cfg.output_root) / "tapnext_chunks"
            out_dir.mkdir(parents=True, exist_ok=True)
            track_npz = out_dir / f"chunk_{dlc_res.chunk_id:05d}_tracks.npz"
            pipeline.write_points_npz(track_npz, pipe_res.points)
            queue_out.put(
                TapChunkResult(
                    chunk_id=dlc_res.chunk_id,
                    start_frame=dlc_res.start_frame,
                    end_frame=dlc_res.end_frame,
                    fps=dlc_res.fps,
                    track_npz_path=str(track_npz),
                    supervision_hits=pipe_res.supervision_hits,
                    windows_run=pipe_res.windows_run,
                )
            )
        except Exception:
            queue_out.put(
                TapChunkResult(
                    chunk_id=dlc_res.chunk_id,
                    start_frame=dlc_res.start_frame,
                    end_frame=dlc_res.end_frame,
                    fps=dlc_res.fps,
                    track_npz_path="",
                    supervision_hits=0,
                    windows_run=0,
                    error=traceback.format_exc(),
                )
            )
    queue_out.put(None)


def merge_chunk_tracks(
    chunk_results: list[TapChunkResult],
    total_frames: int,
    fps: float,
    output_npz: str | Path,
) -> dict[str, Any]:
    if not chunk_results:
        raise ValueError("chunk_results is empty")
    ordered = sorted(chunk_results, key=lambda r: r.chunk_id)
    first = np.load(ordered[0].track_npz_path, allow_pickle=False)
    keypoint_names = [str(v) for v in first["keypoint_names"].tolist()]
    k = len(keypoint_names)

    coords = np.full((total_frames, k, 2), np.nan, dtype=np.float32)
    likelihoods = np.zeros((total_frames, k), dtype=np.float32)
    sources = np.full((total_frames, k), "", dtype="<U16")
    write_counts = np.zeros((total_frames,), dtype=np.int32)

    replaced_rows = 0
    dropped_rows = 0

    for res in ordered:
        data = np.load(res.track_npz_path, allow_pickle=False)
        ts = data["timestamps"].astype(np.float64)
        row_coords = data["coords"].astype(np.float32)
        row_likes = data["likelihoods"].astype(np.float32)
        row_sources = data["sources"]

        frame_idx = np.rint(ts * fps).astype(np.int64) + int(res.start_frame)
        frame_idx = np.clip(frame_idx, 0, total_frames - 1)

        for i, gidx in enumerate(frame_idx):
            new_score = float(np.nanmean(row_likes[i]))
            old_score = float(np.nanmean(likelihoods[gidx]))
            if write_counts[gidx] == 0:
                coords[gidx] = row_coords[i]
                likelihoods[gidx] = row_likes[i]
                sources[gidx] = row_sources[i]
                write_counts[gidx] = 1
                continue
            if new_score >= old_score:
                coords[gidx] = row_coords[i]
                likelihoods[gidx] = row_likes[i]
                sources[gidx] = row_sources[i]
                replaced_rows += 1
            else:
                dropped_rows += 1

    timestamps = np.arange(total_frames, dtype=np.float32) / np.float32(fps)
    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        timestamps=timestamps,
        keypoint_names=np.array(keypoint_names),
        coords=coords,
        likelihoods=likelihoods,
        sources=sources,
    )

    coverage = float((write_counts > 0).mean() * 100.0)
    return {
        "output_npz": str(out),
        "coverage_percent": coverage,
        "replaced_overlap_rows": int(replaced_rows),
        "dropped_overlap_rows": int(dropped_rows),
        "num_chunks_merged": len(ordered),
    }


@app.command()
def main(
    video: str = typer.Option(..., help="Path to long source video"),
    dlc_config: str = typer.Option(..., help="Path to DLC config.yaml"),
    tracker_config: str = typer.Option(..., help="TapNext tracker config YAML"),
    tapnext_ckpt: str = typer.Option(..., help="Path to TAPNext checkpoint"),
    output_dir: str = typer.Option("outputs/slurm_chunked_run", help="Output root directory"),
    output_npz: str = typer.Option(
        "outputs/slurm_chunked_run/final_tracks.npz",
        help="Final merged full-video NPZ",
    ),
    chunk_seconds: float = typer.Option(8.0, min=0.1, help="Chunk duration in seconds"),
    overlap_seconds: float = typer.Option(1.0, min=0.0, help="Overlap duration in seconds"),
    dlc_device: str = typer.Option("cuda:0", help="DLC worker device"),
    tapnext_device: str = typer.Option("cuda:1", help="TapNext worker device"),
    dlc_shuffle: int = typer.Option(1, min=1, help="DLC shuffle to use"),
    dlc_batch_size: int = typer.Option(1, min=1, help="DLC inference batch size"),
    dlc_anchor_stride_seconds: float = typer.Option(
        1.0, help="Anchor stride from DLC CSV. Use <=0 for per-frame anchors"
    ),
    dlc_confidence_floor: float = typer.Option(
        -1.0,
        help="Override DLC anchor confidence floor (use -1 to use tracker config value)",
    ),
    dlc_disable_multithreading: bool = typer.Option(
        True, help="Disable DLC async inference queue to lower memory pressure"
    ),
    tapnext_coord_order: str = typer.Option("yx", help="TapNext coord order: xy or yx"),
    tapnext_sketch_tokens: int = typer.Option(
        0, min=0, help="Experimental TAPNext token sketch size (0 disables)"
    ),
    tapnext_sketch_seed: int = typer.Option(0, help="Sketch random seed"),
    queue_maxsize: int = typer.Option(8, min=1, help="Queue max size"),
    keep_chunks: bool = typer.Option(True, help="Keep intermediate chunk videos"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    _setup_logging(verbose=verbose)
    out_root = Path(output_dir)
    chunks_dir = out_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Splitting video into chunks...")
    fps, frame_count, chunk_jobs = split_video_into_chunks(
        video_path=video,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        chunks_dir=chunks_dir,
    )
    LOGGER.info(
        "Prepared %s chunks from %s frames at %.4f fps",
        len(chunk_jobs),
        frame_count,
        fps,
    )

    ctx = mp.get_context("spawn")
    q1: mp.Queue[ChunkJob | None] = ctx.Queue(maxsize=queue_maxsize)
    q2: mp.Queue[DlcChunkResult | None] = ctx.Queue(maxsize=queue_maxsize)
    q3: mp.Queue[TapChunkResult | None] = ctx.Queue(maxsize=queue_maxsize)

    dlc_cfg = DlcWorkerConfig(
        dlc_config_path=dlc_config,
        tracker_config_path=tracker_config,
        output_root=str(out_root),
        dlc_device=dlc_device,
        dlc_shuffle=dlc_shuffle,
        dlc_batch_size=dlc_batch_size,
        dlc_anchor_stride_seconds=dlc_anchor_stride_seconds,
        dlc_confidence_floor_override=dlc_confidence_floor,
        dlc_disable_multithreading=dlc_disable_multithreading,
    )
    tap_cfg = TapWorkerConfig(
        tracker_config_path=tracker_config,
        tapnext_ckpt=tapnext_ckpt,
        tapnext_device=tapnext_device,
        tapnext_coord_order=tapnext_coord_order,
        tapnext_sketch_tokens=tapnext_sketch_tokens,
        tapnext_sketch_seed=tapnext_sketch_seed,
        output_root=str(out_root),
    )

    dlc_proc = ctx.Process(target=dlc_worker, args=(q1, q2, dlc_cfg), name="dlc-worker")
    tap_proc = ctx.Process(target=tapnext_worker, args=(q2, q3, tap_cfg), name="tapnext-worker")
    dlc_proc.start()
    tap_proc.start()

    for job in chunk_jobs:
        q1.put(job)
    q1.put(None)

    gathered: list[TapChunkResult] = []
    while True:
        try:
            item = q3.get(timeout=5.0)
        except Empty:
            if not dlc_proc.is_alive() and not tap_proc.is_alive():
                break
            continue
        if item is None:
            break
        gathered.append(item)
        LOGGER.info(
            "Received chunk %s result (%s/%s)",
            item.chunk_id,
            len(gathered),
            len(chunk_jobs),
        )

    dlc_proc.join(timeout=30.0)
    tap_proc.join(timeout=30.0)
    if dlc_proc.exitcode not in (0, None):
        raise RuntimeError(f"DLC worker failed with exit code {dlc_proc.exitcode}")
    if tap_proc.exitcode not in (0, None):
        raise RuntimeError(f"TAPNext worker failed with exit code {tap_proc.exitcode}")

    errors = [r for r in gathered if r.error]
    if errors:
        message = "\n\n".join(f"[chunk {r.chunk_id}] {r.error}" for r in errors[:3])
        raise RuntimeError(f"Pipeline failed for {len(errors)} chunks.\n{message}")
    if len(gathered) != len(chunk_jobs):
        raise RuntimeError(
            f"Expected {len(chunk_jobs)} TAP results but received {len(gathered)}"
        )

    LOGGER.info("Merging chunk outputs...")
    merge_stats = merge_chunk_tracks(
        chunk_results=gathered,
        total_frames=frame_count,
        fps=fps,
        output_npz=output_npz,
    )

    manifest_path = out_root / "run_manifest.json"
    manifest = {
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "num_chunks": len(chunk_jobs),
        "dlc_worker": asdict(dlc_cfg),
        "tapnext_worker": asdict(tap_cfg),
        "merge_stats": merge_stats,
        "chunk_jobs": [asdict(j) for j in chunk_jobs],
        "tap_results": [asdict(r) for r in sorted(gathered, key=lambda x: x.chunk_id)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not keep_chunks:
        for job in chunk_jobs:
            try:
                os.remove(job.chunk_path)
            except FileNotFoundError:
                pass

    LOGGER.info("Done. Final NPZ: %s", merge_stats["output_npz"])
    LOGGER.info("Coverage: %.2f%%", merge_stats["coverage_percent"])
    LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    app()
