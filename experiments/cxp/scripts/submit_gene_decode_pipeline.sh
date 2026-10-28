#!/bin/bash
# PARKED — do not submit unless revisiting finetuned SE + gene_decoder_bool:true.
# Submit gene-decode track: emb fit (train2/test2) → transform → export + UMAP → STATE tx train.
#
# Usage: bash experiments/cxp/scripts/submit_gene_decode_pipeline.sh
#   PARTITION_H200=i7_h200 PARTITION_QUICK=scavenger_l4

set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/iridisfs/ddnb/Luke/scFMs/cxp_state}"
CELLX="${CELLXPATHWAY_ROOT:-/iridisfs/ddnb/Luke/cellxpathway}"
PART_H200="${PARTITION_H200:-i7_h200}"
PART_QUICK="${PARTITION_QUICK:-scavenger_l4}"

J_CSV="$(sbatch --partition="${PART_QUICK}" \
  "${CELLX}/scripts/cellxpathway-cxp-v2-emb-finetune-csv.sbatch" | awk '{print $NF}')"
echo "emb CSV + train2/test2 h5ads: ${J_CSV}"

J_FIT="$(sbatch --partition="${PART_H200}" \
  --dependency=afterok:"${J_CSV}" \
  --export=ALL,SKIP_EMB_MATERIALIZE=1 \
  "${STATE_ROOT}/experiments/cxp/emb_finetune_parse/run_emb_finetune_parse.sh" | awk '{print $NF}')"
echo "emb preprocess+fit: ${J_FIT}"

J_XFORM="$(sbatch --partition="${PART_H200}" \
  --dependency=afterok:"${J_FIT}" \
  "${STATE_ROOT}/experiments/cxp/emb_finetune_parse/run_emb_transform_parse.sh" | awk '{print $NF}')"
echo "emb transform (full atlas): ${J_XFORM}"

J_EXPORT="$(sbatch --partition="${PART_QUICK}" \
  --dependency=afterok:"${J_XFORM}" \
  "${CELLX}/scripts/cellxpathway-cxp-v2-export-xstate-finetuned.sbatch" | awk '{print $NF}')"
echo "export X_state finetuned h5ads: ${J_EXPORT}"

J_UMAP="$(sbatch --partition="${PART_QUICK}" \
  --dependency=afterok:"${J_XFORM}" \
  "${CELLX}/scripts/cellxpathway-cxp-v2-embedding-umap.sbatch" | awk '{print $NF}')"
echo "embedding UMAP panel: ${J_UMAP}"

J_TRAIN="$(sbatch --partition="${PART_H200}" \
  --dependency=afterok:"${J_EXPORT}" \
  --export=ALL,CONFIG="${STATE_ROOT}/experiments/cxp/configs/train_X_state_gene_decode.yaml" \
  "${STATE_ROOT}/experiments/cxp/scripts/run_tx_train_cxp.sh" | awk '{print $NF}')"
echo "STATE tx train (gene_decoder true): ${J_TRAIN}"

echo ""
echo "Registry: ${STATE_ROOT}/experiments/cxp/cxp_nested_state_runs.json"
echo "Logs: ${STATE_ROOT}/experiments/cxp/logs/gene_decode-*  and  ${CELLX}/outputs/v2/logs/cxp-v2-*"
