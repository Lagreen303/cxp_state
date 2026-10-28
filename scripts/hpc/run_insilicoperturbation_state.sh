#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=600G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=state_ISP

# Master IN-SILICO PERTURBATION (STATE inference) script
# Usage:
#   sbatch run_ISP.sh [config.yaml]
#   or
#   bash   run_ISP.sh [config.yaml]

set -e
set -o pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_state_repo_env.sh
source "${HPC_DIR}/_state_repo_env.sh"

# -------------------------------------------------------------------
# Config file resolution
# -------------------------------------------------------------------
CONFIG_FILE="${1:-config.yaml}"

if [ ! -f "$CONFIG_FILE" ] && [ -f "${STATE_REPO_ROOT}/$CONFIG_FILE" ]; then
    CONFIG_FILE="${STATE_REPO_ROOT}/$CONFIG_FILE"
elif [ "${CONFIG_FILE#/}" = "$CONFIG_FILE" ]; then
    CONFIG_FILE="${STATE_REPO_ROOT}/$CONFIG_FILE"
fi

# -------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"; }

# -------------------------------------------------------------------
# Sanity check: config exists
# -------------------------------------------------------------------
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

log_info "Starting IN-SILICO PERTURBATION with config: $CONFIG_FILE"
log_info "Repo root: ${STATE_REPO_ROOT}"
log_info "Current working directory: $(pwd)"

# Allow optional skip mechanism if you later add more steps
SKIP_STEPS="${SKIP_STEPS:-}"  # e.g., SKIP_STEPS=02 bash run_ISP.sh
should_skip() {
    local step=$1
    if [[ "$SKIP_STEPS" == *"$step"* ]]; then
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------
# STEP 02: Inference (STATE tx infer)
# -------------------------------------------------------------------
if ! should_skip "02"; then
    log_info "========================================="
    log_info "Step 02: Running in-silico perturbation (STATE inference)"
    log_info "========================================="

    CMD=$(python3 <<PYTHON_SCRIPT
import yaml
import os
import sys
from pathlib import Path

config_file = Path(r"$CONFIG_FILE")
try:
    with config_file.open("r") as f:
        config = yaml.safe_load(f) or {}
except Exception as e:
    print(f"[PYTHON ERROR] Failed to load YAML '{config_file}': {e}", file=sys.stderr)
    sys.exit(1)

# We only care about the 'inference' section here
inf_cfg = (config.get("inference") or {})

# --- required fields that must be present in YAML ---
required_keys = ["output", "model_dir", "checkpoint", "adata", "pert_col"]
missing = [k for k in required_keys if not inf_cfg.get(k)]
if missing:
    print(f"[PYTHON ERROR] Missing required inference keys in YAML: {missing}",
          file=sys.stderr)
    sys.exit(2)

output    = str(inf_cfg["output"])
model_dir = str(inf_cfg["model_dir"]).rstrip("/")   # normalise
adata     = str(inf_cfg["adata"])
pert_col  = str(inf_cfg["pert_col"])
checkpoint = str(inf_cfg["checkpoint"])

# ------------------------------------------------------------------
# CHECKPOINT LOGIC (only thing we resolve):
# If 'checkpoint' is just a filename (no path separator),
# treat it as: <model_dir>/checkpoints/<checkpoint>
# ------------------------------------------------------------------
if os.path.basename(checkpoint) == checkpoint:
    checkpoint = str(Path(model_dir) / "checkpoints" / checkpoint)

# Build CLI args
args = [
    f'--output "{output}"',
    f'--model-dir "{model_dir}"',
    f'--checkpoint "{checkpoint}"',
    f'--adata "{adata}"',
    f'--pert-col "{pert_col}"',
]

# Add optional control_pert if present
if inf_cfg.get("control_pert"):
    args.append(f'--control-pert "{inf_cfg["control_pert"]}"')

cmd = "uv run state tx infer " + " ".join(args)
print(cmd)
PYTHON_SCRIPT
)

    if [ -z "$CMD" ]; then
        log_error "Failed to build inference command."
        exit 1
    fi

    log_info "Running command:"
    echo "  $CMD"

    # Actually run inference
    eval "$CMD"

    log_success "Step 02 (in-silico perturbation) completed successfully"
else
    log_warn "Skipping Step 02 (inference)"
fi

log_info "========================================="
log_success "In-silico perturbation pipeline finished!"
log_info "========================================="
