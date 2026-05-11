#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=230G
#SBATCH --cpus-per-task=24



module load apptainer


apptainer exec -H $PWD \
--bind /iridisfs/ddnb/Luke:/workspace \
--bind /iridisfs/ddnb/Luke:/iridisfs/ddnb/Luke \
--bind /iridisfs/ddnb:/iridisfs/ddnb \
--nv /iridisfs/ddnb/Luke/helical_sb \
python3 "$@"
