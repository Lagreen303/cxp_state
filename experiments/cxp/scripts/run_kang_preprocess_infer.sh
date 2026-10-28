#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --job-name=kang_pre_inf
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/kang-preprocess-infer-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
KANG_DIR="${KANG_DIR:-/iridisfs/ddnb/Luke/cellxpathway/outputs/kang_state/v2_collectri}"
RAW="${KANG_DIR}/kang_x_cxp_raw.h5ad"
TEMPLATE="${KANG_DIR}/kang_x_cxp_template.h5ad"

cd "${STATE_ROOT}"
export UV_OFFLINE="${UV_OFFLINE:-1}"

for req in "${RAW}"; do
  if [[ ! -f "${req}" ]]; then
    echo "[ERROR] Missing ${req}" >&2
    exit 1
  fi
done

echo "[INFO] preprocess_infer: ${RAW} -> ${TEMPLATE}"
uv run state tx preprocess_infer \
  --adata "${RAW}" \
  --output "${TEMPLATE}" \
  --embed-key X_cxp \
  --control-condition control \
  --pert-col cytokine \
  --seed 42

echo "[OK] ${TEMPLATE}"
