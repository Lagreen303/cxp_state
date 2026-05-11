#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=state_predict_cv

set -e
set -o pipefail

BASE_EXP_DIR="/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026"
CV_DIR="${BASE_EXP_DIR}/cross_validation"

for K in 9; do
  RUN_DIR="${CV_DIR}/run_2026_02_05_fold${K}"
  CKPT="${RUN_DIR}/checkpoints/last.ckpt"

  echo "[INFO] Fold ${K}: run dir = ${RUN_DIR}"

  if [ ! -d "${RUN_DIR}" ]; then
    echo "[WARN]   Run directory does not exist, skipping."
    continue
  fi

  if [ ! -f "${CKPT}" ]; then
    echo "[WARN]   Checkpoint not found at ${CKPT}, skipping."
    continue
  fi

  echo "[INFO]   Running prediction with checkpoint ${CKPT}"
  uv run state tx predict \
    --output-dir "${RUN_DIR}" \
    --checkpoint "${CKPT}"
done

echo "[INFO] Cross-validation predictions completed."

