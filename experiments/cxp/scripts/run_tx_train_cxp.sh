#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=600G
#SBATCH --partition=i7_h200
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=cxp_cxp_state_train
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/train-%x-%j.log

# Usage: CONFIG=/abs/path/to.yaml sbatch experiments/cxp/scripts/run_tx_train_cxp.sh

set -euo pipefail
STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
cd "$STATE_ROOT"
export SLURM_SUBMIT_DIR="$STATE_ROOT"
export UV_OFFLINE="${UV_OFFLINE:-1}"
DEFAULT="${STATE_ROOT}/experiments/cxp/configs/train_v2_concat_X_cxp.yaml"
CONFIG="${CONFIG:-$DEFAULT}"
if [[ ! -f "${CONFIG}" ]]; then
  CONFIG="${STATE_ROOT}/experiments/cxp/configs/$(basename "${CONFIG}")"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG not found: ${CONFIG}" >&2
  exit 1
fi
mkdir -p experiments/cxp/logs
bash run_train_state.sh "$CONFIG"
