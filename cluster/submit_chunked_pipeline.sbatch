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

cd /Users/kp/Documents/Playground

VIDEO="/Users/kp/Downloads/01490003_cut.mp4"
DLC_CONFIG="/Users/kp/Documents/Playground/lizard_6pts_bootstrap50-kp-2026-03-11/config.yaml"
TRACKER_CONFIG="/Users/kp/Documents/Playground/configs/lizard_6pts_01490003_cut.yaml"
TAPNEXT_CKPT="/Users/kp/bootstapnext_ckpt.npz"

python /Users/kp/Documents/Playground/scripts/slurm_chunked_dlc_tapnext.py \
  --video "${VIDEO}" \
  --dlc-config "${DLC_CONFIG}" \
  --tracker-config "${TRACKER_CONFIG}" \
  --tapnext-ckpt "${TAPNEXT_CKPT}" \
  --output-dir /Users/kp/Documents/Playground/outputs/slurm_chunked_run \
  --output-npz /Users/kp/Documents/Playground/outputs/slurm_chunked_run/final_tracks.npz \
  --chunk-seconds 3 \
  --overlap-seconds 0 \
  --dlc-device cuda:0 \
  --tapnext-device cuda:1 \
  --dlc-batch-size 1 \
  --dlc-anchor-stride-seconds -1 \
  --dlc-disable-multithreading \
  --keep-chunks
