#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=400G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

uv run state tx predict --output-dir /iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/run_2026_01_29/ --checkpoint /iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/run_2026_01_29/checkpoints/final.ckpt