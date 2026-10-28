#!/bin/bash
# Tmp eval: predict on training-seen cytokines (fewshot test on train1) @ latest ckpt → pathway cell-eval → CV comparison.
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_collectri_seen5_posthoc.sh
#   PARTITION=scavenger_l4 bash experiments/cxp/scripts/submit_cxp_collectri_seen5_posthoc.sh

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
PARTITION="${PARTITION:-scavenger_l4}"
PREDICT_MEM="${PREDICT_MEM:-500G}"
CXP_EXP="${STATE_ROOT}/experiments/cxp"
CONFIG="${CXP_EXP}/configs/cxp_collectri_posthoc_seen5.yaml"
TOML="${CXP_EXP}/tomls/cxp_nested_v2_collectri_seen5.toml"
RESULTS_SUFFIX="_seen5_train_cytokines"
RUN_DIR="${CXP_EXP}/CollecTRI/run_cxp_v2_collectri"

cd "${STATE_ROOT}"
export UV_OFFLINE="${UV_OFFLINE:-1}"

CHECKPOINT_STEP="$(uv run python - <<'PY'
import sys
sys.path.insert(0, "experiments/cxp")
from pathlib import Path
from cxp_collectri_posthoc import load_posthoc_config, resolve_latest_step
cfg = load_posthoc_config("experiments/cxp/configs/cxp_collectri_posthoc_seen5.yaml")
print(resolve_latest_step(Path(cfg["run_dir"])))
PY
)"
echo "[INFO] Latest CollecTRI checkpoint step: ${CHECKPOINT_STEP}"

J_PRED="$(sbatch --partition="${PARTITION}" --mem="${PREDICT_MEM}" \
  --export=ALL,ARM=collectri,CHECKPOINT_STEP="${CHECKPOINT_STEP}",PROFILE=anndata,STATE_ROOT="${STATE_ROOT}",TOML="${TOML}",RESULTS_SUFFIX="${RESULTS_SUFFIX}" \
  "${CXP_EXP}/scripts/run_tx_predict_cxp.sh" | awk '{print $NF}')"
echo "anndata predict (seen5 fewshot test on train1): ${J_PRED}"

J_EVAL="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_PRED}" \
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
echo "Jobs: predict=${J_PRED} celleval=${J_EVAL} plots=${J_PLOTS} compare=${J_CMP}"
echo "Eval dir: ${EVAL_DIR}"
echo "Comparison figures: ${OUT_FIG}/"
echo "Logs: ${CXP_EXP}/logs/"
