#!/bin/bash
# Submit scavenger X_state train segments chained with afterany (resume from last.ckpt each time).
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_xstate_scavenger_chain.sh
#   CHAIN_ROUNDS=5 bash experiments/cxp/scripts/submit_cxp_xstate_scavenger_chain.sh
#
# Each segment: scavenger_l4, 12h, 700G, 1 GPU. Next segment starts when the previous
# ends for any reason (success, timeout, OOM) via Slurm afterany.

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_tx_train_cxp_scavenger.sh"
LOSS_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_cxp_plot_state_train_loss.sh"
CONFIG="${CONFIG:-${STATE_ROOT}/experiments/cxp/configs/train_X_state.yaml}"
PARTITION="${PARTITION:-scavenger_l4}"
CHAIN_ROUNDS="${CHAIN_ROUNDS:-2}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing ${SCRIPT}" >&2
  exit 1
fi

submit_one() {
  local dep="$1"
  if [[ -n "${dep}" ]]; then
    sbatch --partition="${PARTITION}" \
      --dependency="afterany:${dep}" \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${SCRIPT}"
  else
    sbatch --partition="${PARTITION}" \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${SCRIPT}"
  fi
}

echo "[INFO] X_state scavenger chain: ${CHAIN_ROUNDS} segment(s), 12h / 700G each, partition=${PARTITION}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] Resumes from run_cxp_X_state/checkpoints/last.ckpt when present (~34k steps today)"

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
echo "Logs: ${STATE_ROOT}/experiments/cxp/logs/train-cxp_X_state_scav-*.log"
echo "Figure: ${STATE_ROOT}/experiments/cxp/figures/cxp_nested_state_train_loss.pdf"
echo "Tip: CHAIN_ROUNDS=5 schedules more 12h segments without waiting."
