#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=600G
#SBATCH --partition=scavenger_l4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=xstate_allcyto_h5ad
#SBATCH --output=/iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/logs/build-allcyto-donor-xstate-%j.log

# Merge X_state train1+test1 → donor-split train2/test2 (same donors as Collectri allcyto_donor).
set -euo pipefail
STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
cd "$STATE_ROOT"
mkdir -p experiments/cxp/logs
source "${STATE_ROOT}/.venv/bin/activate"
python experiments/cxp/build_allcyto_donor_h5ads.py \
  --train1 "${STATE_ROOT}/experiments/cxp/h5ad/X_state/x_state_train1.h5ad" \
  --test1 "${STATE_ROOT}/experiments/cxp/h5ad/X_state/test1_x_state.h5ad" \
  --out-dir "${STATE_ROOT}/experiments/cxp/h5ad/X_state/h5ad_allcyto_donor" \
  --embed-key X_state \
  --train2-name x_state_allcyto_train2.h5ad \
  --test2-name x_state_allcyto_test2.h5ad
