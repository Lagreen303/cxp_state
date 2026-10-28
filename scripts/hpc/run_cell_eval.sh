#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=400G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#
# Wrapper around the Python metrics script that mirrors STATE's internal
# `state tx predict` evaluation using cell-eval.
#
# Usage:
#   sbatch run_cell_eval.sh path/to/config_infer_GSE178429_IFN_avg.yaml
#   sbatch run_cell_eval.sh path/to/config_infer_GSE178429_IFN_zero.yaml
#

set -e
set -o pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 CONFIG_YAML" >&2
  exit 1
fi

CONFIG_YAML="$1"

if [ ! -f "$CONFIG_YAML" ]; then
  echo "Config YAML not found: $CONFIG_YAML" >&2
  exit 1
fi

# Naïve YAML parsing: extract quoted paths/values from the inference section.
# For the path fields we assume double quotes and grab the text between them.
pred_path=$(awk -F'"' '/output:/ {print $2; exit}' "$CONFIG_YAML")
real_path=$(awk -F'"' '/real:/   {print $2; exit}' "$CONFIG_YAML")
pert_col=$(awk '/pert_col:/ {gsub(/"/, "", $2); print $2; exit}' "$CONFIG_YAML")
control_pert=$(awk '/control_pert:/ {gsub(/"/, "", $2); print $2; exit}' "$CONFIG_YAML")

if [ -z "$pred_path" ] || [ -z "$real_path" ]; then
  echo "Failed to parse 'output' or 'adata' from $CONFIG_YAML" >&2
  exit 1
fi

# Derive output directory: use the experiment run dir and add a
# results_eval_<basename>_genes_all folder so we don't overwrite older runs.
pred_dir=$(dirname "$pred_path")          # .../predictions
run_dir=$(dirname "$pred_dir")           # .../run_2026_02_11_all
pred_base=$(basename "$pred_path" .h5ad) # e.g. GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_predictions
outdir="$run_dir/results_eval_${pred_base}_genes_all"

mkdir -p "$outdir"

echo "Using config:       $CONFIG_YAML"
echo "Predictions (pred): $pred_path"
echo "Real data (real):   $real_path"
echo "Output dir:         $outdir"
echo "control_pert:       $control_pert"
echo "pert_col:           $pert_col"
echo "celltype_col:       cell_type"

uv run --with 'cell-eval,anndata,scipy,pandas,numpy' python analyse_results/state_predict_like_metrics.py \
  --pred "$pred_path" \
  --real "$real_path" \
  --outdir "$outdir" \
  --control-pert "$control_pert" \
  --pert-col "$pert_col" \
  --celltype-col cell_type \
  --profile full

