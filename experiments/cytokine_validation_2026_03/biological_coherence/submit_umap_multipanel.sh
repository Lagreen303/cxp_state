#!/bin/bash
#
# Submit multipanel external validation UMAP to scavenger_l4.
# Output: plots/umaps/umap_multipanel_external_validation.png
#
set -e
cd /iridisfs/ddnb/Luke/scFMs/state
SCRIPT=experiments/cytokine_validation_2026/biological_coherence/make_umap_multipanel.py

sbatch --nodes=1 --partition=scavenger_l4 --cpus-per-task=4 \
       --mem=32G --time=2:00:00 --export=ALL \
       --job-name="umap_multipanel" \
       --wrap="cd /iridisfs/ddnb/Luke/scFMs/state && export PYTHONUNBUFFERED=1 && uv run python ${SCRIPT}"

echo "Submitted umap_multipanel to scavenger_l4"
