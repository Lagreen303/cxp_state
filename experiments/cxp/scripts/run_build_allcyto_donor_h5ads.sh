#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --partition=amd
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cxp_allcyto_h5ad
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/build-allcyto-donor-%j.log

# Build all-cytokine donor-split X_cxp h5ads (train1∪test1 → train2/test2 donors).
# Usage: sbatch experiments/cxp/scripts/run_build_allcyto_donor_h5ads.sh
# Override: sbatch --partition=scavenger_l4 --mem=400G ...

set -euo pipefail
STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
cd "$STATE_ROOT"
mkdir -p experiments/cxp/logs
source "${STATE_ROOT}/.venv/bin/activate"
python experiments/cxp/build_allcyto_donor_h5ads.py
