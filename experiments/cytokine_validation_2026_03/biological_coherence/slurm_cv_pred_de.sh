#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=4:00:00
#SBATCH --job-name=cv_pred_de
#SBATCH --partition=scavenger_l4
#SBATCH --output=cv_pred_de_%j.out
#SBATCH --error=cv_pred_de_%j.err

cd /iridisfs/ddnb/Luke/scFMs/state
uv run python experiments/cytokine_validation_2026/biological_coherence/step7b_cv_pred_de.py
