#!/bin/bash
#SBATCH --job-name=dlc_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-dlc-train-%j.out

set -euo pipefail

# Activate your environment here.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <your-env>

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "${MPLCONFIGDIR}"

cd /Users/kp/Documents/Playground

DLC_CONFIG="/Users/kp/Documents/Playground/lizard_6pts_bootstrap50-kp-2026-03-11/config.yaml"

python /Users/kp/Documents/Playground/scripts/train_dlc.py \
  --dlc-config "${DLC_CONFIG}" \
  --shuffle 2 \
  --trainingsetindex 0 \
  --engine pytorch \
  --device cuda:0 \
  --create-dataset \
  --train \
  --evaluate \
  --plotting \
  --batch-size 8 \
  --save-epochs 5 \
  --epochs 50
