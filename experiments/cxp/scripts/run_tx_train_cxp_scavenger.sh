#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=700G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cxp_X_state_scav
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/train-%x-%j.log

# X_state (or other) STATE tx train on scavenger_l4. Resumes from last.ckpt when present.
#
# Usage:
#   CONFIG=experiments/cxp/configs/train_X_state.yaml sbatch experiments/cxp/scripts/run_tx_train_cxp_scavenger.sh

set -euo pipefail
STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
cd "$STATE_ROOT"
export SLURM_SUBMIT_DIR="$STATE_ROOT"
export UV_OFFLINE="${UV_OFFLINE:-1}"
DEFAULT="${STATE_ROOT}/experiments/cxp/configs/train_X_state.yaml"
CONFIG="${CONFIG:-$DEFAULT}"
if [[ ! -f "${CONFIG}" ]]; then
  CONFIG="${STATE_ROOT}/experiments/cxp/configs/$(basename "${CONFIG}")"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG not found: ${CONFIG}" >&2
  exit 1
fi
mkdir -p experiments/cxp/logs
bash scripts/hpc/run_train_state.sh "$CONFIG"
