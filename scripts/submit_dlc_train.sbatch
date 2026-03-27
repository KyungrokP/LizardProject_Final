#!/bin/bash
#SBATCH --job-name=dlc_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-dlc-train-%j.out

set -euo pipefail

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "${MPLCONFIGDIR}"

# Replace these defaults for your own cluster paths.
PROJECT_ROOT="${PROJECT_ROOT:-/path/to/your/repo}"
DLC_CONFIG="${DLC_CONFIG:-${PROJECT_ROOT}/path/to/your/dlc_project/config.yaml}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

# Compute Canada module stack (edit if your cluster uses different versions).
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load opencv/4.12.0
module load arrow/22.0.0

cd "${PROJECT_ROOT}"

for p in "${PROJECT_ROOT}" "${DLC_CONFIG}"; do
  if [[ ! -e "${p}" ]]; then
    echo "Missing required path: ${p}"
    echo "Edit this sbatch file (or export env vars) before submitting."
    exit 1
  fi
done

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # Use the project virtualenv if available.
  source "${VENV_PATH}/bin/activate"
else
  echo "Warning: venv not found at ${VENV_PATH}; using module python."
fi

python "${PROJECT_ROOT}/scripts/train_dlc.py" \
  --dlc-config "${DLC_CONFIG}" \
  --shuffle 2 \
  --trainingsetindex 0 \
  --device cuda:0 \
  --create-dataset \
  --train \
  --evaluate \
  --plotting \
  --batch-size 8 \
  --save-epochs 5 \
  --epochs 50
