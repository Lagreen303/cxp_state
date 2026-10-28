#!/bin/bash
# Build all-cytokine donor-split h5ads (if needed), then chain 2x60h Collectri STATE
# trains on i7_h200 (afterany resume last.ckpt) + loss figure.
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_collectri_allcyto_donor_i7_chain.sh
#   CHAIN_ROUNDS=2 bash experiments/cxp/scripts/submit_cxp_collectri_allcyto_donor_i7_chain.sh
#   SKIP_BUILD=1 bash ...   # if h5ads already exist

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
PARTITION="${PARTITION:-i7_h200}"
CHAIN_ROUNDS="${CHAIN_ROUNDS:-2}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CONFIG="${STATE_ROOT}/experiments/cxp/configs/train_v2_collectri_X_cxp_allcyto_donor.yaml"
TRAIN_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_tx_train_cxp.sh"
BUILD_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_build_allcyto_donor_h5ads.sh"
LOSS_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_cxp_plot_state_train_loss.sh"
BUILD_PARTITION="${BUILD_PARTITION:-amd}"
BUILD_MEM="${BUILD_MEM:-200G}"
H5AD_DIR="/iridisfs/ddnb/Luke/cellxpathway/outputs/v2/v2_collectri/h5ad_allcyto_donor"
TRAIN_H5AD="${H5AD_DIR}/x_cxp_allcyto_train2.h5ad"
TEST_H5AD="${H5AD_DIR}/x_cxp_allcyto_test2.h5ad"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Missing ${TRAIN_SCRIPT}" >&2
  exit 1
fi

BUILD_DEP=""
if [[ "${SKIP_BUILD}" == "1" ]]; then
  if [[ ! -f "${TRAIN_H5AD}" || ! -f "${TEST_H5AD}" ]]; then
    echo "SKIP_BUILD=1 but h5ads missing under ${H5AD_DIR}" >&2
    exit 1
  fi
  echo "[INFO] SKIP_BUILD=1; using existing ${TRAIN_H5AD}"
else
  if [[ -f "${TRAIN_H5AD}" && -f "${TEST_H5AD}" ]]; then
    echo "[INFO] h5ads already present; skipping build job"
  else
    J_BUILD="$(sbatch --partition="${BUILD_PARTITION}" --mem="${BUILD_MEM}" \
      --export=ALL,STATE_ROOT="${STATE_ROOT}" \
      "${BUILD_SCRIPT}" | awk '{print $NF}')"
    echo "allcyto donor h5ad build: ${J_BUILD}"
    BUILD_DEP="${J_BUILD}"
  fi
fi

submit_train() {
  local dep="$1"
  if [[ -n "${dep}" ]]; then
    sbatch --partition="${PARTITION}" \
      --job-name=cxp_collectri_allcyto \
      --dependency="afterok:${dep}" \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${TRAIN_SCRIPT}"
  else
    sbatch --partition="${PARTITION}" \
      --job-name=cxp_collectri_allcyto \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${TRAIN_SCRIPT}"
  fi
}

echo "[INFO] allcyto donor Collectri chain: ${CHAIN_ROUNDS} x 60h on ${PARTITION}"
echo "[INFO] CONFIG=${CONFIG}"

PREV="${BUILD_DEP}"
JOB_IDS=()
for ((i = 1; i <= CHAIN_ROUNDS; i++)); do
  if [[ -z "${PREV}" ]]; then
    OUT="$(submit_train "")"
  elif [[ "${i}" -eq 1 && -n "${BUILD_DEP}" ]]; then
    # First train waits for successful h5ad build.
    OUT="$(submit_train "${PREV}")"
  else
    # Later segments: afterany so timeout/OOM still resumes last.ckpt.
    OUT="$(sbatch --partition="${PARTITION}" \
      --job-name=cxp_collectri_allcyto \
      --dependency="afterany:${PREV}" \
      --export=ALL,CONFIG="${CONFIG}",STATE_ROOT="${STATE_ROOT}" \
      "${TRAIN_SCRIPT}")"
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
echo "Build dep: ${BUILD_DEP:-none}"
echo "Train chain (job ids): ${JOB_IDS[*]}"
echo "Loss plot: ${J_LOSS}"
echo "Run dir: ${STATE_ROOT}/experiments/cxp/CollecTRI/run_cxp_v2_collectri_allcyto_donor"
echo "Logs: ${STATE_ROOT}/experiments/cxp/logs/train-cxp_collectri_allcyto-*.log"
echo "Build log: ${STATE_ROOT}/experiments/cxp/logs/build-allcyto-donor-*.log"
echo "Figure: ${STATE_ROOT}/experiments/cxp/figures/cxp_nested_state_train_loss.pdf"
