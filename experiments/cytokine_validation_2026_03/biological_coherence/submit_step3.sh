#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --job-name=bio_step3_pred_vs_real

set -e
cd /iridisfs/ddnb/Luke/scFMs/state
export PYTHONUNBUFFERED=1

echo "Step 3: Pred vs real expression ..."
uv run python experiments/cytokine_validation_2026/biological_coherence/step3_pred_vs_real.py
echo "Done."
