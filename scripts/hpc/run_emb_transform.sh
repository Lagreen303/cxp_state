#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=250G
#SBATCH --partition=l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=state_emb_transform
#
# Run STATE **embedding** inference: `state emb transform` from a small YAML file.
#
# Usage:
#   sbatch scripts/hpc/run_emb_transform.sh [path/to/emb_transform.yaml]
#   bash   run_emb_transform.sh [path/to/emb_transform.yaml]
#
# Default config file: emb_transform.yaml next to this script (repo root).
#
# YAML shape (all paths can be absolute or relative to submit cwd):
#   embedding:
#     model_folder: "/path/to/SE-600M-or-checkpoint-dir"
#     input: "/path/to/input.h5ad"
#     output: "/path/to/output.h5ad"          # omit if using only lancedb below
#     # optional:
#     checkpoint: "/path/to/specific.ckpt"   # if omitted, uses latest *.ckpt in model_folder
#     embed_key: "X_state"
#     batch_size: 64
#     protein_embeddings: "/path/to/protein_embeddings.pt"
#     config: "/path/to/override.yaml"       # OmegaConf override; omit to use checkpoint config
#     lancedb: "/path/to/db.lancedb"
#     lancedb_update: false
#     lancedb_batch_size: 1000
#
# See also: examples/emb_transform.yaml

set -e
set -o pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_state_repo_env.sh
source "${HPC_DIR}/_state_repo_env.sh"

CONFIG_FILE="${1:-emb_transform.yaml}"
if [ ! -f "$CONFIG_FILE" ] && [ -f "${STATE_REPO_ROOT}/$CONFIG_FILE" ]; then
    CONFIG_FILE="${STATE_REPO_ROOT}/$CONFIG_FILE"
elif [ "${CONFIG_FILE#/}" = "$CONFIG_FILE" ]; then
    CONFIG_FILE="${STATE_REPO_ROOT}/$CONFIG_FILE"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: embedding YAML not found: $CONFIG_FILE" >&2
    echo "Create it from examples/emb_transform.yaml or pass an explicit path." >&2
    exit 1
fi

CMD=$(CONFIG_FILE="$CONFIG_FILE" python3 <<'PY'
import os
import sys
from pathlib import Path

import yaml

path = Path(os.environ["CONFIG_FILE"])
with path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

emb = cfg.get("embedding") or {}
missing = [k for k in ("model_folder", "input") if not emb.get(k)]
if missing:
    print(f"[ERROR] embedding YAML missing keys: {missing}", file=sys.stderr)
    sys.exit(1)
if not emb.get("output") and not emb.get("lancedb"):
    print("[ERROR] embedding YAML needs at least one of: output, lancedb", file=sys.stderr)
    sys.exit(1)

def q(s: str) -> str:
    return '"' + str(s).replace('"', '\\"') + '"'

parts = [
    "uv run state emb transform",
    f"--model-folder {q(emb['model_folder'])}",
    f"--input {q(emb['input'])}",
]
if emb.get("output"):
    parts.append(f"--output {q(emb['output'])}")

if emb.get("checkpoint"):
    parts.append(f"--checkpoint {q(emb['checkpoint'])}")

if emb.get("embed_key"):
    parts.append(f"--embed-key {q(emb['embed_key'])}")
else:
    parts.append('--embed-key "X_state"')

if emb.get("batch_size") is not None:
    parts.append(f"--batch-size {int(emb['batch_size'])}")

if emb.get("protein_embeddings"):
    parts.append(f"--protein-embeddings {q(emb['protein_embeddings'])}")

if emb.get("config"):
    parts.append(f"--config {q(emb['config'])}")

if emb.get("lancedb"):
    parts.append(f"--lancedb {q(emb['lancedb'])}")
    if emb.get("lancedb_update"):
        parts.append("--lancedb-update")
    lbs = emb.get("lancedb_batch_size")
    if lbs is not None:
        parts.append(f"--lancedb-batch-size {int(lbs)}")

print(" ".join(parts))
PY
) || exit 1

if [ -z "$CMD" ]; then
    echo "ERROR: failed to build emb transform command." >&2
    exit 1
fi

echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Running: $CMD"
eval "$CMD"
