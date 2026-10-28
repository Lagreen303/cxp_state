#!/bin/bash
# Submit stepped test predict (experiments/cxp/cxp_tx_predict.py fork) for both nested CxP arms.
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_stepped_predict.sh
#   CHECKPOINT_STEP=25000 PARTITION=scavenger_l4 bash experiments/cxp/scripts/submit_cxp_stepped_predict.sh

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_tx_predict_cxp.sh"
PARTITION="${PARTITION:-scavenger_l4}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-20000}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing ${SCRIPT}" >&2
  exit 1
fi

submit_arm() {
  local arm="$1"
  local label="$2"
  local job_name="cxp_pred_s${CHECKPOINT_STEP}_${label}"
  sbatch \
    --partition="${PARTITION}" \
    --job-name="${job_name}" \
    --export=ALL,ARM="${arm}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",STATE_ROOT="${STATE_ROOT}" \
    "${SCRIPT}"
}

echo "[INFO] Stepped test predict at step=${CHECKPOINT_STEP} (not last.ckpt final eval)"
echo "[INFO] PARTITION=${PARTITION}"

J1="$(submit_arm x_state X_state)"
J2="$(submit_arm collectri collectri)"
echo "${J1}"
echo "${J2}"

CXP="${STATE_ROOT}/experiments/cxp"
echo ""
echo "Expected results (after jobs complete):"
echo "  ${CXP}/X_state/run_cxp_X_state/eval_step=${CHECKPOINT_STEP}.ckpt/"
echo "  ${CXP}/CollecTRI/run_cxp_v2_collectri/eval_step=${CHECKPOINT_STEP}.ckpt/"
echo ""
echo "Logs: ${CXP}/logs/predict-*.log"
