#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cxp_pred_step
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/predict-%x-%j.log

# Stepped test predict for nested CxP STATE arms (not final last.ckpt eval).
#
# Usage (via submit_cxp_stepped_predict.sh or directly):
#   ARM=x_state|collectri CHECKPOINT_STEP=20000 sbatch experiments/cxp/scripts/run_tx_predict_cxp.sh
#
# Optional:
#   PROFILE=full|minimal|de|anndata  (default: full)
#   PREDICT_ONLY=1                   skip cell-eval metrics
#   sbatch --mem=500G ...            override #SBATCH --mem (seen5 test ~1.15M cells)

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CXP_EXP="${STATE_ROOT}/experiments/cxp"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-20000}"
CHECKPOINT="step=${CHECKPOINT_STEP}.ckpt"
PROFILE="${PROFILE:-full}"
ARM="${ARM:?Set ARM=x_state or ARM=collectri}"

case "${ARM}" in
  x_state)
    RUN_DIR="${CXP_EXP}/X_state/run_cxp_X_state"
    ARM_LABEL="X_state"
    ;;
  collectri)
    RUN_DIR="${CXP_EXP}/CollecTRI/run_cxp_v2_collectri"
    ARM_LABEL="collectri_X_cxp"
    ;;
  *)
    echo "Unknown ARM=${ARM} (expected x_state or collectri)" >&2
    exit 1
    ;;
esac

RESULTS_DIR="${RUN_DIR}/eval_${CHECKPOINT}"
CKPT_PATH="${RUN_DIR}/checkpoints/${CHECKPOINT}"

cd "${STATE_ROOT}"
export SLURM_SUBMIT_DIR="${STATE_ROOT}"
export UV_OFFLINE="${UV_OFFLINE:-1}"
mkdir -p experiments/cxp/logs

echo "[INFO] ARM=${ARM_LABEL} RUN_DIR=${RUN_DIR}"
echo "[INFO] CHECKPOINT=${CHECKPOINT} (stepped test predict, training may still be in progress)"

for req in "${RUN_DIR}/config.yaml" "${RUN_DIR}/data_module.torch" "${CKPT_PATH}"; do
  if [[ ! -e "${req}" ]]; then
    echo "[ERROR] Missing required path: ${req}" >&2
    exit 1
  fi
done

PREDICT_ARGS=(--output-dir "${RUN_DIR}" --checkpoint "${CHECKPOINT}" --profile "${PROFILE}")
if [[ "${PREDICT_ONLY:-0}" == "1" ]]; then
  PREDICT_ARGS+=(--predict-only)
fi
if [[ -n "${TOML:-}" ]]; then
  PREDICT_ARGS+=(--toml "${TOML}")
fi
if [[ -n "${RESULTS_SUFFIX:-}" ]]; then
  PREDICT_ARGS+=(--results-suffix "${RESULTS_SUFFIX}")
fi

echo "[INFO] Running: uv run python experiments/cxp/cxp_tx_predict.py ${PREDICT_ARGS[*]}"
uv run python experiments/cxp/cxp_tx_predict.py "${PREDICT_ARGS[@]}"

MANIFEST="${RESULTS_DIR}/predict_manifest.json"
python3 - <<'PY' "${MANIFEST}" "${ARM_LABEL}" "${CHECKPOINT_STEP}" "${CHECKPOINT}" "${RUN_DIR}" "${RESULTS_DIR}" "${PROFILE}" "${PREDICT_ONLY:-0}" "${SLURM_JOB_ID:-}"
import json
import sys
from datetime import datetime, timezone

manifest_path, arm, step, checkpoint, run_dir, results_dir, profile, predict_only, job_id = sys.argv[1:10]
doc = {
    "kind": "cxp_nested_stepped_test_predict",
    "smoke_test": True,
    "arm": arm,
    "train_step": int(step),
    "checkpoint": checkpoint,
    "run_dir": run_dir,
    "results_dir": results_dir,
    "training_incomplete": (
        "Full STATE training may still be running; this eval uses a stepped checkpoint, not last.ckpt."
    ),
    "profile": profile,
    "predict_only": predict_only == "1",
    "predict_entrypoint": "experiments/cxp/cxp_tx_predict.py",
    "slurm_job_id": job_id or None,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
with open(manifest_path, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
print(f"[INFO] Wrote {manifest_path}")
PY

echo "[INFO] Results: ${RESULTS_DIR}"
