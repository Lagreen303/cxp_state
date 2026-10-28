#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --job-name=cxp_posthoc_plots
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/posthoc-plots-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CONFIG="${CONFIG:-${STATE_ROOT}/experiments/cxp/configs/cxp_collectri_posthoc.yaml}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-}"

cd "${STATE_ROOT}"
export PYTHONPATH="${STATE_ROOT}/experiments/cxp:${PYTHONPATH:-}"
export UV_OFFLINE="${UV_OFFLINE:-1}"

AGG_ARGS=(python experiments/cxp/collectri_posthoc/aggregate_pathway_celleval_celltypes.py --config "${CONFIG}")
if [[ -n "${CHECKPOINT_STEP}" ]]; then
  AGG_ARGS+=(--checkpoint-step "${CHECKPOINT_STEP}")
fi
echo "[INFO] uv run ${AGG_ARGS[*]}"
uv run "${AGG_ARGS[@]}"

RUN_DIR="${STATE_ROOT}/experiments/cxp/CollecTRI/run_cxp_v2_collectri"
STEP="${CHECKPOINT_STEP}"
if [[ -z "${STEP}" ]]; then
  STEP="$(uv run python -c "
import sys
sys.path.insert(0, 'experiments/cxp')
from pathlib import Path
from cxp_collectri_posthoc import load_posthoc_config, resolve_latest_step
cfg = load_posthoc_config('${CONFIG}')
print(resolve_latest_step(Path(cfg['run_dir'])))
")"
fi
EVAL_SUFFIX="$(uv run python -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(open('${CONFIG}'))
print(cfg.get('eval_dir_suffix', '') or '')
")"
AGG_DIR="${RUN_DIR}/eval_step=${STEP}.ckpt${EVAL_SUFFIX}/pathway_celleval/aggregated_celltype_results"

PLOT_ARGS=(
  python experiments/cxp/collectri_posthoc/make_pathway_celleval_heatmaps_and_boxes.py
  --aggregated-dir "${AGG_DIR}"
)
echo "[INFO] uv run ${PLOT_ARGS[*]}"
uv run "${PLOT_ARGS[@]}"
