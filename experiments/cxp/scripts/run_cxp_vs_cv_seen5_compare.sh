#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cxp_vs_cv_seen5
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/cxp-vs-cv-seen5-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
CXP_LONG="${CXP_LONG:?Set CXP_LONG to collectri_celltype_results_long.csv}"
OUT_FIG="${OUT_FIG:-${STATE_ROOT}/experiments/cxp/figures/tmp/seen5_train_cytokines}"
CV_LONG="${STATE_ROOT}/experiments/cytokine_validation_2026_03/cross_validation/aggregated_celltype_results/cv_celltype_results_long.csv"

cd "${STATE_ROOT}"
export PYTHONPATH="${CELLX}:${PYTHONPATH:-}"
export UV_OFFLINE="${UV_OFFLINE:-1}"

uv run python experiments/cxp/plot_cxp_vs_cv_holdout_metrics.py \
  --cxp-long "${CXP_LONG}" \
  --cv-long "${CV_LONG}" \
  --out-dir "${OUT_FIG}" \
  --perts seen5 \
  --cv-label "${CV_LABEL:-CV gene (9-fold aggregate)}"

echo "[INFO] Wrote comparison under ${OUT_FIG}"
