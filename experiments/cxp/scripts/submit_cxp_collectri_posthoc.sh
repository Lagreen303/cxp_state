#!/bin/bash
# CollecTRI posthoc: anndata predict (latest step) → parallel UMAP + pathway cell-eval.
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_collectri_posthoc.sh
#   PARTITION=scavenger_l4 bash experiments/cxp/scripts/submit_cxp_collectri_posthoc.sh

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
PARTITION="${PARTITION:-scavenger_l4}"
CONFIG="${STATE_ROOT}/experiments/cxp/configs/cxp_collectri_posthoc.yaml"
LOSS_SCRIPT="${STATE_ROOT}/experiments/cxp/scripts/run_cxp_plot_state_train_loss.sh"

CHECKPOINT_STEP="$(cd "${STATE_ROOT}" && uv run python - <<'PY'
import sys
sys.path.insert(0, "experiments/cxp")
from pathlib import Path
from cxp_collectri_posthoc import load_posthoc_config, resolve_latest_step
cfg = load_posthoc_config(Path("experiments/cxp/configs/cxp_collectri_posthoc.yaml"))
print(resolve_latest_step(Path(cfg["run_dir"])))
PY
)"

echo "[INFO] Latest CollecTRI checkpoint step: ${CHECKPOINT_STEP}"

J_PRED="$(sbatch --partition="${PARTITION}" \
  --export=ALL,ARM=collectri,CHECKPOINT_STEP="${CHECKPOINT_STEP}",PROFILE=anndata,STATE_ROOT="${STATE_ROOT}" \
  "${STATE_ROOT}/experiments/cxp/scripts/run_tx_predict_cxp.sh" | awk '{print $NF}')"
echo "anndata predict: ${J_PRED}"

J_UMAP="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_PRED}" \
  --export=ALL,CHECKPOINT_STEP="${CHECKPOINT_STEP}",STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}" \
  "${STATE_ROOT}/experiments/cxp/scripts/run_cxp_collectri_posthoc_umap.sh" | awk '{print $NF}')"
echo "pathway UMAP: ${J_UMAP}"

J_EVAL="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_PRED}" \
  --export=ALL,CHECKPOINT_STEP="${CHECKPOINT_STEP}",STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}" \
  "${STATE_ROOT}/experiments/cxp/scripts/run_cxp_collectri_posthoc_celleval.sh" | awk '{print $NF}')"
echo "pathway cell-eval: ${J_EVAL}"

J_PLOTS="$(sbatch --partition="${PARTITION}" \
  --dependency=afterok:"${J_EVAL}" \
  --export=ALL,CHECKPOINT_STEP="${CHECKPOINT_STEP}",STATE_ROOT="${STATE_ROOT}",CONFIG="${CONFIG}" \
  "${STATE_ROOT}/experiments/cxp/scripts/run_cxp_collectri_posthoc_plots.sh" | awk '{print $NF}')"
echo "aggregate + heatmaps/boxes: ${J_PLOTS}"

J_LOSS="$(sbatch --partition="${PARTITION}" \
  --dependency="afterok:${J_PLOTS}" \
  --export=ALL,STATE_ROOT="${STATE_ROOT}" \
  "${LOSS_SCRIPT}" | awk '{print $NF}')"
echo "train loss figure (after posthoc): ${J_LOSS}"

RUN_DIR="${STATE_ROOT}/experiments/cxp/CollecTRI/run_cxp_v2_collectri"
EVAL_DIR="${RUN_DIR}/eval_step=${CHECKPOINT_STEP}.ckpt"
echo ""
echo "After completion:"
echo "  ${EVAL_DIR}/adata_pred.h5ad"
echo "  ${EVAL_DIR}/figures_umap/"
echo "  ${EVAL_DIR}/pathway_celleval/"
echo "  ${EVAL_DIR}/pathway_celleval/aggregated_celltype_results/"
echo "Logs: ${STATE_ROOT}/experiments/cxp/logs/posthoc-*"
