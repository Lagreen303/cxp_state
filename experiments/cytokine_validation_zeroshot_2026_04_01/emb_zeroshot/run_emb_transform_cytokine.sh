#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=250G
#SBATCH --partition=l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12

set -euo pipefail

STATE_ROOT="/iridisfs/ddnb/Luke/scFMs/state"

usage() {
  cat <<'EOF'
Usage:
  sbatch run_emb_transform_cytokine.sh --input <file.h5ad> --output <file.h5ad> [options]

Required:
  --input PATH
  --output PATH

Optional:
  --model-folder PATH         Default: SE-600M snapshot in scratch
  --checkpoint PATH           If omitted, newest *.ckpt in --model-folder is used
  --embed-key NAME            Default: X_state
  --batch-size INT
  --protein-embeddings PATH
  --config PATH               Optional override; default is checkpoint-embedded config
EOF
}

MODEL_FOLDER="/iridisfs/scratch/lag1e24/.cache/huggingface/hub/models--arcinstitute--SE-600M/snapshots/93a4bec2ea5d2da1fa74bfb9181ac714dfcb1198"
CHECKPOINT=""
INPUT_PATH=""
OUTPUT_PATH=""
EMBED_KEY="X_state"
BATCH_SIZE=""
PROTEIN_EMBEDDINGS=""
CONFIG_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-folder) MODEL_FOLDER="${2:-}"; shift 2 ;;
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --input) INPUT_PATH="${2:-}"; shift 2 ;;
    --output) OUTPUT_PATH="${2:-}"; shift 2 ;;
    --embed-key) EMBED_KEY="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --protein-embeddings) PROTEIN_EMBEDDINGS="${2:-}"; shift 2 ;;
    --config) CONFIG_OVERRIDE="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "ERROR: --input and --output are required."
  usage
  exit 2
fi

cd "$STATE_ROOT"

if [[ ! -d "$MODEL_FOLDER" ]]; then
  echo "ERROR: Model folder does not exist: $MODEL_FOLDER"
  exit 2
fi
if [[ ! -f "$INPUT_PATH" ]]; then
  echo "ERROR: Input file does not exist: $INPUT_PATH"
  exit 2
fi

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(ls -1t "$MODEL_FOLDER"/*.ckpt 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: No checkpoint found. Provide --checkpoint or place *.ckpt in: $MODEL_FOLDER"
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

CMD=(uv run state emb transform
  --model-folder "$MODEL_FOLDER"
  --checkpoint "$CHECKPOINT"
  --input "$INPUT_PATH"
  --output "$OUTPUT_PATH"
  --embed-key "$EMBED_KEY"
)

if [[ -n "$CONFIG_OVERRIDE" ]]; then
  CMD+=(--config "$CONFIG_OVERRIDE")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  CMD+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$PROTEIN_EMBEDDINGS" ]]; then
  CMD+=(--protein-embeddings "$PROTEIN_EMBEDDINGS")
fi

echo "Running: state emb transform"
echo "  model_folder: $MODEL_FOLDER"
echo "  checkpoint:   $CHECKPOINT"
echo "  input:        $INPUT_PATH"
echo "  output:       $OUTPUT_PATH"
"${CMD[@]}"
