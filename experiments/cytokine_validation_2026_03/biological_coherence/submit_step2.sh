#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --cpus-per-task=4
#SBATCH --job-name=bio_step2_coherence

set -e
cd /iridisfs/ddnb/Luke/scFMs/state
export PYTHONUNBUFFERED=1

echo "Step 2: Coherence metrics + plots ..."
uv run python experiments/cytokine_validation_2026/biological_coherence/step2_coherence_plots.py
echo "Done."
