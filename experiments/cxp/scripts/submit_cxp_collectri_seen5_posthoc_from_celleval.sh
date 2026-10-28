#!/bin/bash
# Pathway cell-eval → aggregate/heatmaps → CxP vs CV (seen5; predict adatas must exist).
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_collectri_seen5_posthoc_from_celleval.sh
#   CHECKPOINT_STEP=50000 POSTHOC_MEM=500G bash experiments/cxp/scripts/submit_cxp_collectri_seen5_posthoc_from_celleval.sh

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
PARTITION="${PARTITION:-scavenger_l4}"
POSTHOC_MEM="${POSTHOC_MEM:-500G}"
CXP_EXP="${STATE_ROOT}/experiments/cxp"
CONFIG="${CXP_EXP}/configs/cxp_collectri_posthoc_seen5.yaml"
RUN_DIR="${CXP_EXP}/CollecTRI/run_cxp_v2_collectri"
RESULTS_SUFFIX="_seen5_train_cytokines"

cd "${STATE_ROOT}"
export UV_OFFLINE="${UV_OFFLINE:-1}"

CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"
if [[ -z "${CHECKPOINT_STEP}" ]]; then
  CHECKPOINT_STEP="$(uv run python - <<'PY'
import sys
sys.path.insert(0, "experiments/cxp")
from pathlib import Path
from cxp_collectri_posthoc import load_posthoc_config, resolve_latest_step
cfg = load_posthoc_config("experiments/cxp/configs/cxp_collectri_posthoc_seen5.yaml")
print(resolve_latest_step(Path(cfg["run_dir"])))
PY
)"
fi
echo "[INFO] CollecTRI checkpoint step: ${CHECKPOINT_STEP}"

J_EVAL="$(sbatch --partition="${PARTITION}" --mem="${POSTHOC_MEM}" \
  --export=ALL,CHECKPOINT_STEP="${CHECKPOINT_STEP}",STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}" \
  "${CXP_EXP}/scripts/run_cxp_collectri_posthoc_celleval.sh" | awk '{print $NF}')"
echo "pathway cell-eval: ${J_EVAL}"

J_PLOTS="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_EVAL}" \
  --export=ALL,CHECKPOINT_STEP="${CHECKPOINT_STEP}",STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}" \
  "${CXP_EXP}/scripts/run_cxp_collectri_posthoc_plots.sh" | awk '{print $NF}')"
echo "aggregate + heatmaps: ${J_PLOTS}"

EVAL_DIR="${RUN_DIR}/eval_step=${CHECKPOINT_STEP}.ckpt${RESULTS_SUFFIX}"
CXP_LONG="${EVAL_DIR}/pathway_celleval/aggregated_celltype_results/collectri_celltype_results_long.csv"
OUT_FIG="${CXP_EXP}/figures/tmp/seen5_train_cytokines"

J_CMP="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_PLOTS}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}",CHECKPOINT_STEP="${CHECKPOINT_STEP}",CXP_LONG="${CXP_LONG}",OUT_FIG="${OUT_FIG}" \
  "${CXP_EXP}/scripts/run_cxp_vs_cv_seen5_compare.sh" | awk '{print $NF}')"
echo "CxP vs CV comparison plots: ${J_CMP}"

echo ""
echo "Jobs: celleval=${J_EVAL} plots=${J_PLOTS} compare=${J_CMP}"
echo "Eval dir: ${EVAL_DIR}"
echo "Comparison figures: ${OUT_FIG}/"
