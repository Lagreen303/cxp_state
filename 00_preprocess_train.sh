#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --mem=250G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=state_process_cytokine

uv run state tx preprocess_train \
    --adata /iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/Parse_10M_PBMC_cytokines.h5ad \
    --output /iridisfs/ddnb/Luke/data/parse-10M-90cytokine/parse-10M-90cytokines_processed_state.h5ad \
    --num_hvgs 2000 \