#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=200G
#SBATCH --partition=quad_h200,a100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --job-name=state_emb_all
#SBATCH --export=ALL

set -euo pipefail

ROOT="/iridisfs/ddnb/Luke/scFMs/state"
CFG="/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_zeroshot_2026_04_01/full_train/configs/config_all.yaml"

cd "$ROOT"
bash run_train_state.sh "$CFG"

