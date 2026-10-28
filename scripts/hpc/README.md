# HPC / Slurm wrappers

All cluster drivers for this fork live here. **`setup_uv_env.sh` stays at the repo root** — source it before interactive work, or let these scripts load it via `_state_repo_env.sh`.

```bash
cd /path/to/cxp_state
source setup_uv_env.sh
uv sync

sbatch scripts/hpc/run_train_state.sh config.yaml
sbatch scripts/hpc/run_cell_eval.sh path/to/infer_config.yaml
```

CxP-specific train/posthoc chains are under **`experiments/cxp/scripts/`** (they call `scripts/hpc/run_train_state.sh` where needed).

See **[docs/HPC_SLURM_WRAPPERS.md](../docs/HPC_SLURM_WRAPPERS.md)** for the full script index.
