#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=600G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bio_step1_atlas_de

set -e
cd /iridisfs/ddnb/Luke/scFMs/state
export PYTHONUNBUFFERED=1

echo "Step 1: Atlas DE ..."
uv run python experiments/cytokine_validation_2026/biological_coherence/step1_atlas_de.py
echo "Done."
