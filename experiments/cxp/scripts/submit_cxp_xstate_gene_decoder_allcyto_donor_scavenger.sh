#!/bin/bash
# Build X_state all-cytokine donor-split h5ads (if needed), then chain gene-decoder STATE train.
#
# Same donors / val layout as run_cxp_v2_collectri_allcyto_donor (train2_test2.json).
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_xstate_gene_decoder_allcyto_donor_scavenger.sh
#   CHAIN_ROUNDS=10 bash experiments/cxp/scripts/submit_cxp_xstate_gene_decoder_allcyto_donor_scavenger.sh
#   SKIP_BUILD=1 bash ...   # h5ads already built

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_tx_train_cxp_scavenger.sh"
BUILD_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_build_allcyto_donor_xstate_h5ads.sh"
CONFIG="${STATE_ROOT}/experiments/cxp/configs/train_X_state_gene_decoder_allcyto_donor.yaml"
PARTITION="${PARTITION:-scavenger_l4}"
CHAIN_ROUNDS="${CHAIN_ROUNDS:-10}"
SKIP_BUILD="${SKIP_BUILD:-0}"

H5AD_DIR="${STATE_ROOT}/experiments/cxp/h5ad/X_state/h5ad_allcyto_donor"
TRAIN_H5AD="${H5AD_DIR}/x_state_allcyto_train2.h5ad"
TEST_H5AD="${H5AD_DIR}/x_state_allcyto_test2.h5ad"

DEP_ARGS=()
if [[ "${SKIP_BUILD}" != "1" && ( ! -f "${TRAIN_H5AD}" || ! -f "${TEST_H5AD}" ) ]]; then
  J_BUILD="$(sbatch --partition=scavenger_l4 "${BUILD_SCRIPT}" | awk '{print $NF}')"
  echo "xstate allcyto donor h5ad build: ${J_BUILD}"
  DEP_ARGS=(--dependency=afterok:"${J_BUILD}")
elif [[ ! -f "${TRAIN_H5AD}" || ! -f "${TEST_H5AD}" ]]; then
  echo "Missing ${TRAIN_H5AD} or ${TEST_H5AD} (set SKIP_BUILD=0 to build)" >&2
  exit 1
else
  echo "[INFO] Using existing h5ads under ${H5AD_DIR}"
fi

submit_one() {
  local dep="$1"
  if [[ -n "${dep}" ]]; then
    sbatch --partition="${PARTITION}" \
      --job-name=cxp_xstate_gdec_allcyto \
      --dependency="afterany:${dep}" \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${SCRIPT}"
  else
    sbatch --partition="${PARTITION}" \
      --job-name=cxp_xstate_gdec_allcyto \
      "${DEP_ARGS[@]}" \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${SCRIPT}"
  fi
}

echo "[INFO] X_state gene_decoder allcyto_donor: ${CHAIN_ROUNDS} x 12h on ${PARTITION}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] Fresh run dir (no resume from legacy gene_decoder):"
echo "       ${STATE_ROOT}/experiments/cxp/X_state_gene_decoder_allcyto_donor/run_cxp_X_state_gene_decoder_allcyto_donor"

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

echo ""
echo "Train chain (afterany): ${JOB_IDS[*]}"
echo "Logs: ${STATE_ROOT}/experiments/cxp/logs/train-cxp_X_state_scav-*.log"
echo "Split: ${STATE_ROOT}/../cellxpathway/configs/v2/train2_test2.json"
