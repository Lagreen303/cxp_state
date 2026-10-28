#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --job-name=cxp_vs_cv_s5_tm
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/cxp-vs-cv-seen5-train-matched-%j.log

# CxP pathway (training-seen seen5) vs CV gene metrics from fold1 re-predict
# (same five cytokines were in fold1 training).

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
CXP_LONG="${CXP_LONG:?Set CXP_LONG to collectri_celltype_results_long.csv}"
OUT_FIG="${OUT_FIG:-${STATE_ROOT}/experiments/cxp/figures/tmp/seen5_train_cytokines_cv_fold1_matched}"
CV_LONG="${CV_LONG:-${STATE_ROOT}/experiments/cytokine_validation_2026_03/cross_validation/aggregated_celltype_results/cv_seen5_train_cytokines_fold1_long.csv}"

cd "${STATE_ROOT}"
export PYTHONPATH="${CELLX}:${PYTHONPATH:-}"
export UV_OFFLINE="${UV_OFFLINE:-1}"
# Avoid home-dir quota on compute nodes (uv cache).
export UV_CACHE_DIR="${UV_CACHE_DIR:-${STATE_ROOT}/.uv_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STATE_ROOT}/.cache}"
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}"

uv run python experiments/cxp/plot_cxp_vs_cv_holdout_metrics.py \
  --cxp-long "${CXP_LONG}" \
  --cv-long "${CV_LONG}" \
  --out-dir "${OUT_FIG}" \
  --perts seen5 \
  --cv-label "${CV_LABEL:-CV gene (fold1, train-matched)}"

echo "[INFO] Wrote training-matched comparison under ${OUT_FIG}"
