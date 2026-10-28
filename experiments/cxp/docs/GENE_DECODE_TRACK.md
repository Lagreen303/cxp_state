# Gene-decode CxP–STATE track

**Parked (2026-03-13):** SE finetune + gene-decode Slurm chain cancelled. Arc/STATE practice is **frozen** zeroshot embeddings for tx; use **`nested_v2_embed`** only.

Parallel to **nested_v2_embed**. Machine-readable registry: [`cxp_nested_state_runs.json`](../cxp_nested_state_runs.json).

## Tracks

| track_id | gene_decoder | Embeddings | Predict |
|----------|--------------|------------|---------|
| `nested_v2_embed` | false | Zeroshot SE (`state_embed_h5ad` in pipeline.yaml) | `experiments/cxp/cxp_tx_predict.py` |
| `gene_decode_xstate` | true | Finetuned SE on train2/test2 split, transform on full HVG atlas | `uv run state tx predict` |

Do **not** compare `val_loss` across tracks (embed-only vs embed+decoder).

## SE finetune alignment with VAE

- **Train:** `manifest_train2.parquet` (same as v2 panel VAE fit).
- **Val:** `manifest_test2.parquet` (donor holdout, same as VAE eval).
- **Expression:** `cellxpathway/outputs/v2/shared/parse_atlas_state_hvg2000.h5ad`.
- **Transform:** all cells in that atlas → `parse_atlas_state_hvg2000_emb_finetuned.h5ad`.

## Slurm chain

```bash
bash /iridisfs/ddnb/Luke/scFMs/cxp_state/experiments/cxp/scripts/submit_gene_decode_pipeline.sh
```

Logs: `experiments/cxp/logs/gene_decode-*`, `cellxpathway/outputs/v2/logs/`.
