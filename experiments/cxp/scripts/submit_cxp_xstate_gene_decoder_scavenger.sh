#!/bin/bash
# X_state with gene_decoder_bool:true on scavenger (12h segments, resume last.ckpt).
#
# Usage:
#   bash experiments/cxp/scripts/submit_cxp_xstate_gene_decoder_scavenger.sh
#   CHAIN_ROUNDS=5 bash experiments/cxp/scripts/submit_cxp_xstate_gene_decoder_scavenger.sh

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
export CONFIG="${STATE_ROOT}/experiments/cxp/configs/train_X_state_gene_decoder.yaml"
export CHAIN_ROUNDS="${CHAIN_ROUNDS:-4}"
bash "${STATE_ROOT}/experiments/cxp/scripts/submit_cxp_xstate_scavenger_chain.sh"
