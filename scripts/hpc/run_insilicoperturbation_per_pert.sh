#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=400G
#SBATCH --partition=l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=state_ISP

# Master IN-SILICO PERTURBATION (STATE inference) script (BATCH MODE)
# Iterates over a directory of template .h5ad files and writes prediction .h5ads
#
# Usage:
#   sbatch run_ISP_dir.sh [config.yaml]
#   or
#   bash   run_ISP_dir.sh [config.yaml]
#
# YAML requirements (inference section):
#   inference:
#     templates_dir: "/path/to/templates_dir"     # directory of input templates (.h5ad)
#     predictions_dir: "/path/to/predictions_dir" # output directory for predictions (.h5ad)
#     model_dir: "/path/to/model/run_dir"
#     checkpoint: "final.ckpt"                    # or full path; if filename -> <model_dir>/checkpoints/<checkpoint>
#     pert_col: "pert_gene_test"
#     control_pert: "control"                     # optional (only if your state CLI supports it)

set -e
set -o pipefail

# -------------------------------------------------------------------
# Locate script directory (works under SLURM and direct bash)
# -------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
elif [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(pwd)"
fi

cd "$SCRIPT_DIR"

# -------------------------------------------------------------------
# UV environment setup (your existing script)
# -------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/setup_uv_env.sh" ]; then
    source "$SCRIPT_DIR/setup_uv_env.sh"
else
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/uv/cache}"
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRIPT_DIR/uv/pythons}"
    export UV_TOOL_DIR="${UV_TOOL_DIR:-$SCRIPT_DIR/uv/tools}"
    mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" 2>/dev/null || true
fi

# -------------------------------------------------------------------
# Config file resolution
# -------------------------------------------------------------------
CONFIG_FILE="${1:-config.yaml}"

if [ ! -f "$CONFIG_FILE" ] && [ -f "$SCRIPT_DIR/$CONFIG_FILE" ]; then
    CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
elif [ "${CONFIG_FILE#/}" = "$CONFIG_FILE" ]; then
    CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
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

log_info "Starting IN-SILICO PERTURBATION (DIR MODE) with config: $CONFIG_FILE"
log_info "Script directory: $SCRIPT_DIR"
log_info "Current working directory: $(pwd)"

# -------------------------------------------------------------------
# Optional skip mechanism
# -------------------------------------------------------------------
SKIP_STEPS="${SKIP_STEPS:-}"
should_skip() {
    local step=$1
    if [[ "$SKIP_STEPS" == *"$step"* ]]; then
        return 0
    fi
    return 1
}

# -------------------------------------------------------------------
# STEP 02: Inference over directory (STATE tx infer)
# -------------------------------------------------------------------
if ! should_skip "02"; then
    log_info "========================================="
    log_info "Step 02: Running in-silico perturbation over templates directory"
    log_info "========================================="

    # Build and run the whole loop in Python (so we don't fight with YAML parsing in bash)
    python3 <<PYTHON_SCRIPT
import os
import sys
import yaml
import shlex
import subprocess
from pathlib import Path

CONFIG_FILE = Path(r"$CONFIG_FILE")

def die(msg: str, code: int = 2):
    print(f"[PYTHON ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

try:
    config = yaml.safe_load(CONFIG_FILE.read_text()) or {}
except Exception as e:
    die(f"Failed to load YAML '{CONFIG_FILE}': {e}", 1)

inf_cfg = (config.get("inference") or {})

# Required keys for dir-mode
required = ["templates_dir", "predictions_dir", "model_dir", "checkpoint", "pert_col"]
missing = [k for k in required if not inf_cfg.get(k)]
if missing:
    die(f"Missing required inference keys in YAML: {missing}")

templates_dir = Path(str(inf_cfg["templates_dir"])).expanduser()
pred_dir      = Path(str(inf_cfg["predictions_dir"])).expanduser()
model_dir     = Path(str(inf_cfg["model_dir"])).expanduser()
checkpoint    = str(inf_cfg["checkpoint"])
pert_col      = str(inf_cfg["pert_col"])

control_pert = inf_cfg.get("control_pert", None)

if not templates_dir.exists():
    die(f"templates_dir does not exist: {templates_dir}")
if not templates_dir.is_dir():
    die(f"templates_dir is not a directory: {templates_dir}")

pred_dir.mkdir(parents=True, exist_ok=True)

# Resolve checkpoint:
# If it's just a filename (no path separators), interpret as <model_dir>/checkpoints/<checkpoint>
if os.path.basename(checkpoint) == checkpoint:
    checkpoint_path = model_dir / "checkpoints" / checkpoint
else:
    checkpoint_path = Path(checkpoint).expanduser()

if not checkpoint_path.exists():
    die(f"Checkpoint not found: {checkpoint_path}")

# Collect templates
templates = sorted(templates_dir.glob("*.h5ad"))
if len(templates) == 0:
    die(f"No .h5ad templates found in: {templates_dir}")

print(f"[INFO] templates_dir: {templates_dir}", flush=True)
print(f"[INFO] predictions_dir: {pred_dir}", flush=True)
print(f"[INFO] Found {len(templates):,} template(s).", flush=True)
print(f"[INFO] model_dir: {model_dir}", flush=True)
print(f"[INFO] checkpoint: {checkpoint_path}", flush=True)
print(f"[INFO] pert_col: {pert_col}", flush=True)

# Build per-template infer command
# Output naming: <template_stem>__prediction.h5ad
failed = 0

for i, tpl in enumerate(templates, start=1):
    out_path = pred_dir / f"{tpl.stem}__prediction.h5ad"

    # Skip if already exists (handy for restarts)
    if out_path.exists():
        print(f"[SKIP] ({i}/{len(templates)}) Output exists: {out_path.name}", flush=True)
        continue

    cmd = [
        "uv", "run", "state", "tx", "infer",
        "--output", str(out_path),
        "--model-dir", str(model_dir),
        "--checkpoint", str(checkpoint_path),
        "--adata", str(tpl),
        "--pert-col", pert_col,
    ]

    # Only append control_pert if set
    if control_pert is not None and str(control_pert).strip() != "":
        cmd += ["--control-pert", str(control_pert)]

    print(f"\n[RUN] ({i}/{len(templates)}) {tpl.name}", flush=True)
    print("      ->", out_path.name, flush=True)
    print("      CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        failed += 1
        print(f"[FAIL] Template failed: {tpl} (exit={e.returncode})", file=sys.stderr, flush=True)
        # continue running the rest (don't kill a 2000-template run)
        continue

print(f"\n[INFO] Completed. Failed templates: {failed}", flush=True)
if failed > 0:
    sys.exit(3)
PYTHON_SCRIPT

    log_success "Step 02 (dir inference) finished"
else
    log_warn "Skipping Step 02 (inference)"
fi

log_info "========================================="
log_success "In-silico perturbation pipeline finished!"
log_info "========================================="
