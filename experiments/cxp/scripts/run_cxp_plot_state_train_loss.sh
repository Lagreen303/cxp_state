#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cxp_train_loss_fig
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/plot-train-loss-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
PY="${CELLX}/.venv/bin/python"
SCRIPT="${CELLX}/scripts/cxp_v2_plot_state_train_loss.py"

cd "${CELLX}"
export PYTHONPATH="${CELLX}:${PYTHONPATH:-}"

echo "[INFO] ${PY} ${SCRIPT} --state-root ${STATE_ROOT}"
"${PY}" "${SCRIPT}" --state-root "${STATE_ROOT}"
