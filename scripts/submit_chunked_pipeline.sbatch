#!/bin/bash
#SBATCH --job-name=dlc_tapnext_chunked
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Optional for MPS-heavy PyTorch environments:
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Activate your environment here.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your-env>

# Replace these defaults for your own cluster paths.
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/your/repo}"
VIDEO="${VIDEO:-/path/to/your/video.mp4}"
DLC_CONFIG="${DLC_CONFIG:-${PROJECT_ROOT}/path/to/your/dlc_project/config.yaml}"
TRACKER_CONFIG="${TRACKER_CONFIG:-${PROJECT_ROOT}/configs/default.yaml}"
TAPNEXT_CKPT="${TAPNEXT_CKPT:-/path/to/your/tapnext_checkpoint.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/slurm_chunked_run}"
OUTPUT_NPZ="${OUTPUT_NPZ:-${OUTPUT_DIR}/final_tracks.npz}"

cd "${PROJECT_ROOT}"

for p in "${PROJECT_ROOT}" "${VIDEO}" "${DLC_CONFIG}" "${TRACKER_CONFIG}" "${TAPNEXT_CKPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "Missing required path: ${p}"
    echo "Edit this sbatch file (or export env vars) before submitting."
    exit 1
  fi
done

python "${PROJECT_ROOT}/scripts/slurm_chunked_dlc_tapnext.py" \
  --video "${VIDEO}" \
  --dlc-config "${DLC_CONFIG}" \
  --tracker-config "${TRACKER_CONFIG}" \
  --tapnext-ckpt "${TAPNEXT_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-npz "${OUTPUT_NPZ}" \
  --chunk-seconds 3 \
  --overlap-seconds 0 \
  --dlc-device cuda:0 \
  --tapnext-device cuda:1 \
  --dlc-batch-size 1 \
  --dlc-anchor-stride-seconds -1 \
  --dlc-disable-multithreading \
  --keep-chunks
