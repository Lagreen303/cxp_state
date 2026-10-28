# CxP fork: signed embedding targets (`X_cxp`, `X_state`)

This tree (`cxp_state`) is a STATE fork for **Cell×Pathway (CxP) latents**, not raw gene counts.

## Output activation (no ReLU on CxP latents)

`StateTransitionPerturbationModel` (`src/state/tx/models/state_transition.py`) sets `self.relu` from `output_activation` (default `auto`):

| `embed_key` | `output_activation` | `self.relu` |
|-------------|---------------------|-------------|
| `X_hvg` or `None` | `auto` | `ReLU` (counts-like target) |
| `X_cxp`, `X_state`, … | `auto` | **`Identity`** (signed latents) |
| any | `relu` | `ReLU` |
| any | `none` | `Identity` |

All CxP training configs under `experiments/cxp/configs/` use `model: state_sm` with `embed_key: X_cxp` or `X_state`. **Standard `state tx infer` from this repo does not clamp negative latent predictions.**

Optional override: `training.output_activation: none|relu|auto` in the Hydra/YAML stack if you need to force behaviour for ablations.

## What still uses ReLU elsewhere

- **`pseudobulk` model** — not used by CxP configs (`state_sm` only).
- **scGPT / embedding loaders** — unrelated to CxP `tx` heads.
- **Gene decoders** — may apply non-negativity on **decoded gene space** when `gene_decoder_bool: true`; that is separate from the transition head on `X_cxp` / `X_state`.

## Kang external infer

Use `experiments/cxp/run_tx_infer_kang_external.sh` (or `submit_kang_collectri_state_infer.sh`). Do **not** use a separate infer patch; the fork behaviour is in `state_transition.py`.

Legacy checkpoints trained before this fork change should still load with `embed_key` in the checkpoint hparams so `auto` selects `Identity`.
