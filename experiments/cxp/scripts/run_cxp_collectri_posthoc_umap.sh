#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cxp_posthoc_umap
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/posthoc-umap-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
CONFIG="${CONFIG:-${STATE_ROOT}/experiments/cxp/configs/cxp_collectri_posthoc.yaml}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
PY="${CELLX}/.venv/bin/python"

cd "${STATE_ROOT}"
export PYTHONPATH="${CELLX}:${STATE_ROOT}/experiments/cxp:${PYTHONPATH:-}"

ARGS=("${PY}" experiments/cxp/cxp_collectri_pathway_umap.py --config "${CONFIG}")
if [[ -n "${CHECKPOINT_STEP}" ]]; then
  ARGS+=(--checkpoint-step "${CHECKPOINT_STEP}")
fi

echo "[INFO] Running: ${ARGS[*]}"
"${ARGS[@]}"
