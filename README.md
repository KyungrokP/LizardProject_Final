# Hybrid Object Motion Tracking with DeepLabCut and TAPNext

This repository provides a hybrid tracking pipeline for animal/object motion:
- DeepLabCut (DLC) predicts supervised keypoint anchors.
- TAPNext performs short-window tracking (5-second model limit).
- Fusion combines TAPNext trajectories with DLC anchors for better stability.

The workflow is split into:
- Local workflow: labeling, debugging, and quick CPU runs.
- Cluster workflow: GPU training and large-scale inference.

## Core Scripts

- `scripts/init_dlc_project.py`: create a new DLC project.
- `scripts/train_dlc.py`: create dataset, train DLC, evaluate DLC.
- `scripts/export_dlc_queries.py`: run DLC and export anchor NPZ queries.
- `scripts/run_pipeline.py`: run TAPNext + DLC fusion on one video.
- `scripts/render_overlay.py`: render tracked points on video.
- `scripts/slurm_chunked_dlc_tapnext.py`: chunked queue pipeline (DLC -> TAPNext -> merge).
- `scripts/submit_dlc_train.sbatch`: Slurm template for DLC training.
- `scripts/submit_chunked_pipeline.sbatch`: Slurm template for chunked inference.

## Local Workflow (Labeling, Quick Runs, CPU)

### 1) Create a DLC project

```bash
python scripts/init_dlc_project.py \
  --project-name my_tracking \
  --experimenter your_name \
  --video /absolute/path/to/video.mp4 \
  --working-directory .
```

### 2) Extract and label frames

Use DLC API (or GUI) to extract frames, then label:

```bash
python - <<'PY'
import deeplabcut as dlc
CONFIG="/absolute/path/to/config.yaml"
dlc.extract_frames(CONFIG, mode="automatic", algo="kmeans", userfeedback=False)
dlc.label_frames(CONFIG)
PY
```

If the DLC GUI closes unexpectedly, labeling from a specific frame folder with Napari often helps:

```bash
python - <<'PY'
import deeplabcut as dlc
import napari
CONFIG="/absolute/path/to/config.yaml"
dlc.label_frames(CONFIG, image_folder="your_video_folder_name")
napari.run()
PY
```

### 3) Train DLC locally (quick CPU run)

```bash
python scripts/train_dlc.py \
  --dlc-config /absolute/path/to/config.yaml \
  --shuffle 2 \
  --device cpu \
  --create-dataset \
  --train \
  --evaluate \
  --epochs 50 \
  --batch-size 8 \
  --save-epochs 5
```

### 4) Export DLC anchors for fusion

3-second anchors (recommended for sparse supervision):

```bash
python scripts/export_dlc_queries.py \
  --dlc-config /absolute/path/to/config.yaml \
  --video /absolute/path/to/video.mp4 \
  --mode sampled_3s \
  --tracker-config configs/default.yaml \
  --out-npz outputs/dlc_queries_3s.npz
```

Per-frame anchors (stronger supervision, slower):

```bash
python scripts/export_dlc_queries.py \
  --dlc-config /absolute/path/to/config.yaml \
  --video /absolute/path/to/video.mp4 \
  --mode full_video \
  --fps 59.94 \
  --tracker-config configs/default.yaml \
  --out-npz outputs/dlc_queries_full.npz
```

### 5) Run TAPNext + DLC fusion and render output

```bash
python scripts/run_pipeline.py \
  --video /absolute/path/to/video.mp4 \
  --duration 30 \
  --config configs/default.yaml \
  --supervision-npz outputs/dlc_queries_3s.npz \
  --tapnext-backend local_torch \
  --tapnext-ckpt /absolute/path/to/tapnext_checkpoint.npz \
  --tapnext-device cpu \
  --output-npz outputs/tracking_output.npz
```

```bash
python scripts/render_overlay.py \
  --video /absolute/path/to/video.mp4 \
  --tracking-npz outputs/tracking_output.npz \
  --output-video outputs/tracking_overlay.mp4 \
  --color-by-source
```

## Cluster Workflow (GPU Training and Large-Scale Inference)

### 1) Label locally, then transfer only required DLC data

Transfer these items to cluster:
- DLC project config (for example `config_01520001_only.yaml`)
- `labeled-data/<video_name>/`
- video file(s) used for training/inference
- `scripts/train_dlc.py` and `scripts/submit_dlc_train.sbatch`

Optional compression for faster transfer:

```bash
tar -czf /tmp/dlc_train_bundle.tar.gz \
  /absolute/path/to/dlc_project/config_01520001_only.yaml \
  /absolute/path/to/dlc_project/labeled-data/01520001 \
  /absolute/path/to/dlc_project/videos/01520001.mp4
```

### 2) Train DLC on cluster GPU

Edit placeholders in `scripts/submit_dlc_train.sbatch`:
- `PROJECT_ROOT`
- `DLC_CONFIG`
- `VENV_PATH` (optional)

Then submit:

```bash
sbatch scripts/submit_dlc_train.sbatch
```

Training logs are written to:
- `slurm_logs/slurm-dlc-train-<jobid>.out`

### 3) Run chunked DLC -> TAPNext pipeline on cluster

Edit placeholders in `scripts/submit_chunked_pipeline.sbatch`:
- `PROJECT_ROOT`
- `VIDEO`
- `DLC_CONFIG`
- `TRACKER_CONFIG`
- `TAPNEXT_CKPT`

Then submit:

```bash
sbatch scripts/submit_chunked_pipeline.sbatch
```

## Queue Pipeline Procedure (Multiprocessing)

`scripts/slurm_chunked_dlc_tapnext.py` follows this procedure:

1. Split long video into overlapping chunks.
2. Queue chunk metadata (`Queue 1`).
3. DLC worker reads chunks and runs `deeplabcut.analyze_videos(...)`.
4. DLC worker converts predictions to anchor NPZ and pushes metadata (`Queue 2`).
5. TAPNext worker reads anchors from `Queue 2`, tracks chunk, saves NPZ (`Queue 3`).
6. Merge worker stitches chunks, drops overlap duplicates, writes final full-video NPZ.

  
