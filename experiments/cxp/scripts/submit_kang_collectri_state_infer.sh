#!/bin/bash
# Chain: Kang encode (cellxpathway) → preprocess_infer → tx infer (last.ckpt).
#
# Usage:
#   bash experiments/cxp/scripts/submit_kang_collectri_state_infer.sh
#   ENCODE_JOB=12345 bash experiments/cxp/scripts/submit_kang_collectri_state_infer.sh
#   SMOKE=1 bash experiments/cxp/scripts/submit_kang_collectri_state_infer.sh  # 500-cell encode smoke

set -euo pipefail

CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
PARTITION="${PARTITION:-scavenger_l4}"

if [[ -n "${ENCODE_JOB:-}" ]]; then
  ENC_DEP="${ENCODE_JOB}"
else
  if [[ "${SMOKE:-0}" == "1" ]]; then
    ENC_DEP="$(SMOKE=1 bash "${CELLX}/scripts/kang_state/submit_encode_kang_v2_collectri.sh" | awk -F': ' '/^encode_kang/{print $2}')"
  else
    ENC_DEP="$(bash "${CELLX}/scripts/kang_state/submit_encode_kang_v2_collectri.sh" | awk -F': ' '/^encode_kang/{print $2}')"
  fi
fi

J_PRE="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${ENC_DEP}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}" \
  "${STATE_ROOT}/experiments/cxp/scripts/run_kang_preprocess_infer.sh" | awk '{print $NF}')"
echo "kang preprocess_infer: ${J_PRE}"

J_INF="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_PRE}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}",CHECKPOINT="${CHECKPOINT:-}" \
  "${STATE_ROOT}/experiments/cxp/scripts/run_tx_infer_kang_external.sh" | awk '{print $NF}')"
echo "kang tx infer: ${J_INF}"

echo ""
echo "After completion:"
echo "  ${CELLX}/outputs/kang_state/v2_collectri/kang_x_cxp_simulated.h5ad"
echo "Posthoc: bash ${STATE_ROOT}/experiments/cxp/scripts/submit_kang_collectri_posthoc.sh ${J_INF}"
