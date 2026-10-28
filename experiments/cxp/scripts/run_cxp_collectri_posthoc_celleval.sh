#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=12
#SBATCH --job-name=cxp_posthoc_eval
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/posthoc-celleval-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
CONFIG="${CONFIG:-${STATE_ROOT}/experiments/cxp/configs/cxp_collectri_posthoc.yaml}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"

cd "${STATE_ROOT}"
export PYTHONPATH="${CELLX}:${STATE_ROOT}/experiments/cxp:${PYTHONPATH:-}"
export UV_OFFLINE="${UV_OFFLINE:-1}"

ARGS=(python experiments/cxp/cxp_collectri_pathway_celleval.py --config "${CONFIG}")
if [[ -n "${CHECKPOINT_STEP}" ]]; then
  ARGS+=(--checkpoint-step "${CHECKPOINT_STEP}")
fi

echo "[INFO] Running: uv run ${ARGS[*]}"
uv run "${ARGS[@]}"
