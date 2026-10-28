#!/bin/bash
# Kang infer posthoc: pathway UMAP + cell-eval + aggregate plots.
#
# Usage:
#   bash experiments/cxp/scripts/submit_kang_collectri_posthoc.sh
#   bash experiments/cxp/scripts/submit_kang_collectri_posthoc.sh 2000123   # after infer job id

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
PARTITION="${PARTITION:-scavenger_l4}"
CONFIG="${STATE_ROOT}/experiments/cxp/configs/cxp_collectri_posthoc_kang.yaml"
INF_DEP="${1:-}"

DEP_ARGS=()
if [[ -n "${INF_DEP}" ]]; then
  DEP_ARGS=(--dependency=afterok:"${INF_DEP}")
fi

J_UMAP="$(sbatch --partition="${PARTITION}" \
  "${DEP_ARGS[@]}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}",CHECKPOINT_STEP=0 \
  "${STATE_ROOT}/experiments/cxp/scripts/run_cxp_collectri_posthoc_umap.sh" | awk '{print $NF}')"
echo "kang pathway UMAP: ${J_UMAP}"

J_EVAL="$(sbatch --partition="${PARTITION}" \
  "${DEP_ARGS[@]}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}",CHECKPOINT_STEP=0 \
  "${STATE_ROOT}/experiments/cxp/scripts/run_cxp_collectri_posthoc_celleval.sh" | awk '{print $NF}')"
echo "kang pathway cell-eval: ${J_EVAL}"

J_PLOTS="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_EVAL}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}",CHECKPOINT_STEP=0 \
  "${STATE_ROOT}/experiments/cxp/scripts/run_cxp_collectri_posthoc_plots.sh" | awk '{print $NF}')"
echo "kang aggregate plots: ${J_PLOTS}"

J_DIAG="$(sbatch --partition="${PARTITION}" \
  "${DEP_ARGS[@]}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}",CHECKPOINT_STEP=0 \
  "${STATE_ROOT}/experiments/cxp/scripts/run_cxp_collectri_pathway_metric_diagnostics.sh" | awk '{print $NF}')"
echo "kang pathway metric diagnostics: ${J_DIAG}"

OUT="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}/outputs/kang_state/v2_collectri/posthoc"
echo ""
echo "Outputs under: ${OUT}/figures_umap/ and pathway_celleval/"
