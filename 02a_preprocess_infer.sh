#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=400G
#SBATCH --partition=scavenger_l4
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

uv run state tx preprocess_infer \
  --adata  /iridisfs/ddnb/Luke/data/GSE96583_Kang/GSE96583_batch2_IFNbeta_aligned_avg.h5ad \
  --output /iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/templates/GSE96583_batch2_IFNbeta_aligned_avg_processed.h5ad \
  --control-condition "control" \
  --pert-col "cytokine" \
  --seed 42