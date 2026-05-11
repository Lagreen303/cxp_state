#!/bin/bash
# Submit all biological coherence steps as SLURM jobs.
# Step 1 (atlas DE) must finish before step 2 (which reads atlas_de_results.csv).
# Steps 3 and 4 are independent and can run in parallel.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting step 1: Atlas DE ..."
JOB1=$(sbatch --parsable "$DIR/submit_step1.sh")
echo "  Job $JOB1"

echo "Submitting step 2: Coherence metrics + plots (depends on step 1) ..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$DIR/submit_step2.sh")
echo "  Job $JOB2"

echo "Submitting step 3: Pred vs real expression (independent) ..."
JOB3=$(sbatch --parsable "$DIR/submit_step3.sh")
echo "  Job $JOB3"

echo "Submitting step 4: UMAPs (independent) ..."
JOB4=$(sbatch --parsable "$DIR/submit_step4.sh")
echo "  Job $JOB4"

echo ""
echo "All jobs submitted:"
echo "  Step 1 (atlas DE):        $JOB1"
echo "  Step 2 (coherence/plots): $JOB2 (after $JOB1)"
echo "  Step 3 (pred vs real):    $JOB3"
echo "  Step 4 (UMAPs):           $JOB4"
