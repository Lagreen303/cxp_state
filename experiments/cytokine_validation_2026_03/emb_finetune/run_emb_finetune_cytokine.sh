#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=12
#SBATCH --time=60:00:00
#SBATCH --partition=a100
#SBATCH --export=ALL
#SBATCH --gpus=2

set -e
STATE_ROOT="/iridisfs/ddnb/Luke/scFMs/state"
EMB_FINETUNE="/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/emb_finetune"
cd "$STATE_ROOT"

# Paths (absolute so sbatch cwd does not matter)
TRAIN_CSV="${EMB_FINETUNE}/cytokine_atlas_train.csv"
VAL_CSV="${EMB_FINETUNE}/cytokine_atlas_val.csv"
OUTPUT_DIR="${EMB_FINETUNE}/preprocess_output"
CONFIG="${EMB_FINETUNE}/config_emb_finetune_cytokine.yaml"

# Protein embeddings from SE-600M model (gene_symbol -> [5120] tensor)
ALL_EMBEDDINGS="/iridisfs/scratch/lag1e24/.cache/huggingface/hub/models--arcinstitute--SE-600M/snapshots/93a4bec2ea5d2da1fa74bfb9181ac714dfcb1198/protein_embeddings.pt"
if [ ! -f "$ALL_EMBEDDINGS" ]; then
    echo "ERROR: Embeddings file not found: $ALL_EMBEDDINGS"
    exit 1
fi

# Step 1: Preprocess - creates ds_emb_mapping, valid_genes_masks, updates config
echo "Step 1: Preprocessing cytokine atlas..."
uv run state emb preprocess \
  --profile-name cytokine_atlas \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --all-embeddings "$ALL_EMBEDDINGS" \
  --config-file "$CONFIG"

# Preprocess updates the config file - merge output into our config
# (Preprocess adds embeddings.cytokine_atlas and dataset.cytokine_atlas)
# The preprocess overwrites/merges - we need to ensure our experiment.init_from is preserved
# Preprocess only adds embeddings and dataset blocks; our config has experiment already

# Step 2: Finetune
echo "Step 2: Finetuning SE model on cytokine atlas..."
uv run state emb fit --conf "$CONFIG" \
  embeddings.current=cytokine_atlas \
  dataset.current=cytokine_atlas

echo "Done. Checkpoints saved to experiment.checkpoint.path in config."
