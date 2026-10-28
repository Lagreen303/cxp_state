#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cxp_path_diag
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/pathway-metric-diag-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
CONFIG="${CONFIG:-${STATE_ROOT}/experiments/cxp/configs/cxp_collectri_posthoc.yaml}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
PY="${CELLX}/.venv/bin/python"
SCRIPT="${STATE_ROOT}/experiments/cxp/collectri_posthoc/pathway_metric_diagnostics.py"

cd "${STATE_ROOT}"
export PYTHONPATH="${CELLX}:${STATE_ROOT}/experiments/cxp:${PYTHONPATH:-}"

ARGS=("${PY}" "${SCRIPT}" --config "${CONFIG}")
if [[ -n "${CHECKPOINT_STEP}" ]]; then
  ARGS+=(--checkpoint-step "${CHECKPOINT_STEP}")
fi
echo "[INFO] ${ARGS[*]}"
"${ARGS[@]}"
