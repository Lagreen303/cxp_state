#!/bin/bash
# All-cytokine donor-val Collectri STATE train on scavenger_l4 (12h segments, afterany).
# h5ads must already exist (build completed). Resumes last.ckpt each segment.
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_collectri_allcyto_donor_scavenger_chain.sh
#   CHAIN_ROUNDS=10 bash experiments/cxp/scripts/submit_cxp_collectri_allcyto_donor_scavenger_chain.sh
#
# Default CHAIN_ROUNDS=10 ≈ 120h wall (matches prior 2×60h i7 plan); scavenger max is 12h/job.

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_tx_train_cxp_scavenger.sh"
LOSS_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_cxp_plot_state_train_loss.sh"
CONFIG="${CONFIG:-${STATE_ROOT}/experiments/cxp/configs/train_v2_collectri_X_cxp_allcyto_donor.yaml}"
PARTITION="${PARTITION:-scavenger_l4}"
CHAIN_ROUNDS="${CHAIN_ROUNDS:-10}"
H5AD_DIR="/iridisfs/ddnb/Luke/cellxpathway/outputs/v2/v2_collectri/h5ad_allcyto_donor"
TRAIN_H5AD="${H5AD_DIR}/x_cxp_allcyto_train2.h5ad"
TEST_H5AD="${H5AD_DIR}/x_cxp_allcyto_test2.h5ad"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_H5AD}" || ! -f "${TEST_H5AD}" ]]; then
  echo "Missing allcyto donor h5ads under ${H5AD_DIR}" >&2
  exit 1
fi

submit_one() {
  local dep="$1"
  if [[ -n "${dep}" ]]; then
    sbatch --partition="${PARTITION}" \
      --job-name=cxp_collectri_allcyto_scav \
      --dependency="afterany:${dep}" \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${SCRIPT}"
  else
    sbatch --partition="${PARTITION}" \
      --job-name=cxp_collectri_allcyto_scav \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${SCRIPT}"
  fi
}

echo "[INFO] allcyto donor Collectri scavenger chain: ${CHAIN_ROUNDS} x 12h on ${PARTITION}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] Resumes from run_cxp_v2_collectri_allcyto_donor/checkpoints/last.ckpt when present"

PREV=""
JOB_IDS=()
for ((i = 1; i <= CHAIN_ROUNDS; i++)); do
  if [[ -z "${PREV}" ]]; then
    OUT="$(submit_one "")"
  else
    OUT="$(submit_one "${PREV}")"
  fi
  echo "${OUT}"
  JID="$(echo "${OUT}" | awk '{print $NF}')"
  JOB_IDS+=("${JID}")
  PREV="${JID}"
done

LAST="${JOB_IDS[-1]}"
J_LOSS="$(sbatch --partition=scavenger_l4 \
  --dependency="afterany:${LAST}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}" \
  "${LOSS_SCRIPT}" | awk '{print $NF}')"
echo "train loss figure (afterany last segment): ${J_LOSS}"

echo ""
echo "Chain (afterany): ${JOB_IDS[*]}"
echo "Loss plot: ${J_LOSS}"
echo "Run dir: ${STATE_ROOT}/experiments/cxp/CollecTRI/run_cxp_v2_collectri_allcyto_donor"
echo "Logs: ${STATE_ROOT}/experiments/cxp/logs/train-cxp_collectri_allcyto_scav-*.log"
echo "Figure: ${STATE_ROOT}/experiments/cxp/figures/cxp_nested_state_train_loss.pdf"
echo "Tip: CHAIN_ROUNDS=N to change segment count (12h each)."
