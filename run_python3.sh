#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=1200G
#SBATCH --partition=quad_h200
#SBATCH --export=ALL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

module load apptainer
apptainer exec -H $PWD \
--bind /iridisfs/scratch/lag1e24:/iridisfs/scratch/lag1e24 \
--bind /iridisfs/home/lag1e24:/iridisfs/home/lag1e24 \
--bind /iridisfs/ddnb/Luke:/workspace \
--bind /iridisfs/ddnb/:/iridisfs/ddnb/ \
--nv /iridisfs/ddnb/Luke/helical_sb \
python3 "$@"
