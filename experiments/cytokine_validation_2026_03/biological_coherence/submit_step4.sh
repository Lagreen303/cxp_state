#!/bin/bash
#
# Submit each UMAP task as a separate SLURM job.
# All jobs are independent — no dependencies between them.
#
set -e
cd /iridisfs/ddnb/Luke/scFMs/state
SCRIPT=experiments/cytokine_validation_2026/biological_coherence/step4_umaps.py

submit() {
    local TASK=$1
    local MEM=$2
    local TIME=$3
    sbatch --nodes=1 --partition=scavenger_l4 --cpus-per-task=8 \
           --mem="${MEM}" --time="${TIME}" --export=ALL \
           --job-name="umap_${TASK}" \
           --wrap="cd /iridisfs/ddnb/Luke/scFMs/state && export PYTHONUNBUFFERED=1 && uv run python ${SCRIPT} --task ${TASK}"
    echo "  Submitted: ${TASK}  (mem=${MEM}, time=${TIME})"
}

echo "Submitting UMAP jobs ..."

# Large jobs: atlas ~10M cells, cv_predictions ~14M cells
submit atlas           600G 12:00:00
submit cv_predictions  600G 12:00:00

# External datasets are much smaller (~tens of thousands of cells)
submit GSE96583_real   100G 4:00:00
submit GSE96583_pred   100G 4:00:00
submit GSE96583_joint  100G 4:00:00
submit GSE178429_real  100G 4:00:00
submit GSE178429_pred  100G 4:00:00
submit GSE178429_joint 100G 4:00:00

echo "All 8 UMAP jobs submitted."
