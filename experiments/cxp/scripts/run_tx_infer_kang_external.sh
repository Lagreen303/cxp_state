#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=kang_tx_infer
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/kang-tx-infer-%j.log

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
MODEL_DIR="${MODEL_DIR:-${STATE_ROOT}/experiments/cxp/CollecTRI/run_cxp_v2_collectri}"
KANG_DIR="${KANG_DIR:-/iridisfs/ddnb/Luke/cellxpathway/outputs/kang_state/v2_collectri}"
CHECKPOINT="${CHECKPOINT:-${MODEL_DIR}/checkpoints/last.ckpt}"
TEMPLATE="${KANG_DIR}/kang_x_cxp_template.h5ad"
OUT="${KANG_DIR}/kang_x_cxp_simulated.h5ad"

cd "${STATE_ROOT}"
export UV_OFFLINE="${UV_OFFLINE:-1}"
mkdir -p experiments/cxp/logs "${KANG_DIR}/posthoc"

for req in "${TEMPLATE}" "${CHECKPOINT}" "${MODEL_DIR}/config.yaml"; do
  if [[ ! -e "${req}" ]]; then
    echo "[ERROR] Missing ${req}" >&2
    exit 1
  fi
done

uv run python - <<'PY' "${MODEL_DIR}/pert_onehot_map.pt"
import sys, torch
m = torch.load(sys.argv[1], weights_only=False)
if "IFN-beta" not in m:
    raise SystemExit("IFN-beta missing from pert_onehot_map.pt")
print("IFN-beta present in pert_onehot_map.pt")
PY

echo "[INFO] tx infer checkpoint=${CHECKPOINT}"
uv run state tx infer \
  --model-dir "${MODEL_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --adata "${TEMPLATE}" \
  --output "${OUT}" \
  --embed-key X_cxp \
  --pert-col cytokine \
  --control-pert control \
  --celltype-col cell_type \
  --seed 42

echo "[OK] ${OUT}"
