# CxP STATE experiments (code only in git)

Slurm drivers and Python posthoc live here. **Checkpoints, h5ad, plots, and `run_*` training dirs stay local** (see repo `.gitignore`).

## Layout

| Path | Contents |
|------|-----------|
| `configs/` | Hydra/YAML templates (`train_*.yaml`, posthoc configs) |
| `tomls/` | Nested CV TOML task definitions |
| `scripts/` | `run_*`, `submit_*`, build h5ad helpers |
| `scripts/env_cxp.sh` | `STATE_ROOT` default → repo root (`cxp_state`) |
| `docs/` | Track notes (e.g. gene decode) |
| `*.py` (root of `cxp/`) | Posthoc / plotting entrypoints |

## Quick start

```bash
cd /path/to/cxp_state
source experiments/cxp/scripts/env_cxp.sh
uv sync

# Example: CollecTRI X_cxp train chain
bash experiments/cxp/scripts/submit_cxp_collectri_allcyto_donor_scavenger_chain.sh

# Kang external validation (infer uses Identity on X_cxp — see docs/CXP_FORK.md)
bash experiments/cxp/scripts/submit_kang_collectri_state_infer.sh
```

Override `STATE_ROOT`, `CONFIG`, or `PARTITION` in the environment as needed.
