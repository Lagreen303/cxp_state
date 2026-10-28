#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=400G
#SBATCH --partition=l4
#SBATCH --export=ALL
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --job-name=state_train

# Master STATE TRAINING script (STATE train)
# Usage:
#   sbatch scripts/hpc/run_train_state.sh [config.yaml]
#   bash   scripts/hpc/run_train_state.sh [config.yaml]

set -e
set -o pipefail

# -------------------------------------------------------------------
# Locate STATE repo root (this script's directory). Do not use SLURM_SUBMIT_DIR:
# CxP jobs are often sbatch'd from cellxpathway, which would send uv to the wrong project.
# -------------------------------------------------------------------
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(pwd)"
fi

# shellcheck source=_state_repo_env.sh
source "${SCRIPT_DIR}/_state_repo_env.sh"

# -------------------------------------------------------------------
# UV environment setup (via setup_uv_env.sh at repo root)
# -------------------------------------------------------------------
if [ ! -f "${STATE_REPO_ROOT}/setup_uv_env.sh" ]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-${STATE_REPO_ROOT}/uv/cache}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${STATE_REPO_ROOT}/uv/pythons}"
    export UV_TOOL_DIR="${UV_TOOL_DIR:-${STATE_REPO_ROOT}/uv/tools}"
    mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" 2>/dev/null || true
fi

# -------------------------------------------------------------------
# Config file resolution
# -------------------------------------------------------------------
CONFIG_FILE="${1:-config.yaml}"

# If config is relative, resolve from repo root
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

log_info "Starting STATE TRAINING with config: $CONFIG_FILE"
log_info "Repo root: ${STATE_REPO_ROOT}"
log_info "Current working directory: $(pwd)"

# Optional skip mechanism
SKIP_STEPS="${SKIP_STEPS:-}"  # e.g., SKIP_STEPS=01 bash run_train_state.sh
should_skip() {
    local step=$1
    if [[ "$SKIP_STEPS" == *"$step"* ]]; then
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------
# STEP 01: Training (STATE tx train)
# -------------------------------------------------------------------
if ! should_skip "01"; then
    log_info "========================================="
    log_info "Step 01: Running STATE training"
    log_info "========================================="

    CMD=$(python3 <<PYTHON_SCRIPT
import yaml
import sys
from pathlib import Path

config_file = Path(r"$CONFIG_FILE")
try:
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
except Exception as e:
    print(f"[PYTHON ERROR] Failed to load YAML '{config_file}': {e}", file=sys.stderr)
    sys.exit(1)

train_cfg = (config.get("training") or {})
data_cfg  = (train_cfg.get("data") or {})
hp_cfg    = (train_cfg.get("training") or {})
wandb_cfg = (train_cfg.get("wandb") or {})
out_cfg   = (train_cfg.get("output") or {})

# --- required keys (based on your YAML + Hydra config) ---
required = {
    "data.toml_config_path": data_cfg.get("toml_config_path"),
    "data.batch_col": data_cfg.get("batch_col"),
    "data.pert_col": data_cfg.get("pert_col"),
    "data.cell_type_key": data_cfg.get("cell_type_key"),
    "data.control_pert": data_cfg.get("control_pert"),
    "output.output_dir": out_cfg.get("output_dir"),
    "output.name": out_cfg.get("name"),
    "model": train_cfg.get("model"),
}
missing = [k for k, v in required.items() if v in (None, "", [])]
if missing:
    print(f"[PYTHON ERROR] Missing required training keys in YAML: {missing}", file=sys.stderr)
    sys.exit(2)

overrides = []

# ---- Output/run naming ----
output_dir = str(out_cfg["output_dir"])
name       = str(out_cfg["name"])
overrides.append(f'output_dir="{output_dir}"')
overrides.append(f'name="{name}"')

# ---- WandB / logging ----
use_wandb = bool(wandb_cfg.get("project"))
overrides.append(f"use_wandb={str(use_wandb).lower()}")

if wandb_cfg.get("project"):
    overrides.append(f'wandb.project="{wandb_cfg["project"]}"')
if wandb_cfg.get("entity"):
    overrides.append(f'wandb.entity="{wandb_cfg["entity"]}"')

tags_val = wandb_cfg.get("tags", None)
if tags_val is not None:
    try:
        parsed = yaml.safe_load(tags_val) if isinstance(tags_val, str) else tags_val
        if isinstance(parsed, (list, tuple)) and parsed:
            tags_expr = "[" + ",".join(str(t) for t in parsed) + "]"
            overrides.append(f"wandb.tags={tags_expr}")
    except Exception:
        overrides.append(f'wandb.tags="{tags_val}"')

# ---- Data kwargs (matches configs/data/perturbation.yaml) ----
toml_path = data_cfg["toml_config_path"]
overrides.append(f'data.kwargs.toml_config_path="{toml_path}"')

for key in [
    "num_workers",
    "batch_col",
    "pert_col",
    "cell_type_key",
    "control_pert",
    "embed_key",
    "output_space",
    "perturbation_features_file",
]:
    if key in data_cfg and data_cfg[key] is not None:
        val = data_cfg[key]
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
            overrides.append(f"data.kwargs.{key}={val}")
        else:
            overrides.append(f'data.kwargs.{key}="{val}"')

# ---- Training hyperparameters (matches configs/training/default.yaml) ----
for key, val in (hp_cfg or {}).items():
    if val is None:
        continue
    if isinstance(val, (int, float, bool)) or (isinstance(val, str) and val.isdigit()):
        overrides.append(f"training.{key}={val}")
    else:
        overrides.append(f'training.{key}="{val}"')

# ---- Model group ----
model_name = train_cfg.get("model")
overrides.append(f"model={model_name}")

# ---- Optional: pretrained checkpoint (init_from for finetuning) ----
init_from = train_cfg.get("init_from")
if init_from:
    overrides.append(f'model.kwargs.init_from="{init_from}"')

model_kwargs = train_cfg.get("model_kwargs") or {}
for key, val in model_kwargs.items():
    if val is None:
        continue
    if isinstance(val, bool):
        overrides.append(f"model.kwargs.{key}={str(val).lower()}")
    elif isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
        overrides.append(f"model.kwargs.{key}={val}")
    else:
        overrides.append(f'model.kwargs.{key}="{val}"')

cmd = "uv run state tx train " + " ".join(overrides)
print(cmd)
PYTHON_SCRIPT
)

    if [ -z "$CMD" ]; then
        log_error "Failed to build training command."
        exit 1
    fi

    log_info "Running command:"
    echo "  $CMD"

    # Actually run training
    eval "$CMD"

    log_success "Step 01 (training) completed successfully"
else
    log_warn "Skipping Step 01 (training)"
fi

log_info "========================================="
log_success "STATE training pipeline finished!"
log_info "========================================="
