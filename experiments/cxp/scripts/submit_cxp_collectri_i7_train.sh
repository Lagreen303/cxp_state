#!/bin/bash
# Resume CollecTRI nested STATE train on i7_h200 + refresh loss figure when the job ends.
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_collectri_i7_train.sh

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
PARTITION="${PARTITION:-i7_h200}"
CONFIG="${STATE_ROOT}/experiments/cxp/configs/train_v2_collectri_X_cxp.yaml"
TRAIN_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_tx_train_cxp.sh"
LOSS_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_cxp_plot_state_train_loss.sh"

J_TRAIN="$(sbatch --partition="${PARTITION}" \
  --job-name=cxp_collectri_train \
  --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
  "${TRAIN_SCRIPT}" | awk '{print $NF}')"
echo "CollecTRI train (i7_h200, resume last.ckpt): ${J_TRAIN}"

J_LOSS="$(sbatch --partition=scavenger_l4 \
  --dependency="afterany:${J_TRAIN}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}" \
  "${LOSS_SCRIPT}" | awk '{print $NF}')"
echo "train loss figure (afterany train): ${J_LOSS}"

echo ""
echo "Run dir: ${STATE_ROOT}/experiments/cxp/CollecTRI/run_cxp_v2_collectri"
echo "Logs: ${STATE_ROOT}/experiments/cxp/logs/train-cxp_collectri_train-*.log"
echo "Figure: ${STATE_ROOT}/experiments/cxp/figures/cxp_nested_state_train_loss.pdf"
