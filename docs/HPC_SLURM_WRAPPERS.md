# HPC / SLURM wrappers (this fork)

These shell scripts live under **`scripts/hpc/`** (repo root names are one-line forwards). They wrap the upstream [`state` CLI](https://github.com/ArcInstitute/state) for **University of Southampton Iridis** (or similar) SLURM partitions. They are **not** part of the upstream Arc Institute package; treat paths (`#SBATCH` partitions, Apptainer binds, `config.yaml` entries) as templates you must adapt to your cluster account.

## Environment: `uv` and `setup_uv_env.sh`

1. Install [`uv`](https://docs.astral.sh/uv/) on the login or compute node.
2. From the repo root: `source setup_uv_env.sh`  
   This sets `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_TOOL_DIR`, and `UV_PROJECT_ENVIRONMENT` under the repo (or whatever you override in the environment). Edit the file if your site requires different filesystem locations.
3. Install / sync the project: `uv sync` (or `uv run state --help` to trigger env creation).

Scripts that embed training or inference (`run_train_state.sh`, `run_insilicoperturbation_state.sh`, etc.) source `setup_uv_env.sh` when present.

## Wrapper overview

| Script | Role |
|--------|------|
| `setup_uv_env.sh` (repo root) | Export project-local `uv` paths; **source** before interactive work. |
| `scripts/hpc/00_preprocess_train.sh` | SLURM job: example `state tx preprocess_train` with **hardcoded** input/output `.h5ad` paths—copy and edit for your data. |
| `02a_preprocess_infer.sh` | SLURM job: example `state tx preprocess_infer` for a control template—copy and edit paths. |
| `01_run_train_state_config.sh` | SLURM job: reads `config.yaml` (or path argument), builds `uv run state tx train ...` overrides via embedded Python, then runs training. Lighter than `run_train_state.sh`. |
| `run_train_state.sh` | SLURM-oriented driver: sources `setup_uv_env.sh`, resolves `config.yaml`, builds a richer Hydra override set (including optional `init_from`, wandb tags), runs `uv run state tx train`. Supports `SKIP_STEPS=01` to skip training. |
| `02_run_inference_state_config.sh` | SLURm job: reads YAML `inference` block, builds `uv run state tx infer ...`. |
| `run_insilicoperturbation_state.sh` | Same pattern as `run_train_state.sh` but for the in-silico perturbation pipeline (YAML-driven). |
| `run_predict_state.sh` | Minimal example: `state tx predict` with fixed paths—template only. |
| `run_emb_transform.sh` | Reads YAML key **`embedding`** (see `examples/emb_transform.yaml`), runs `uv run state emb transform` (writes `.h5ad` and/or LanceDB). Default file: `emb_transform.yaml`. |
| `run_python3.sh` | SLURM + Apptainer: runs **host** `python3` inside `/iridisfs/ddnb/Luke/helical_sb` with GPU and filesystem binds—**not** the `state` venv unless you invoke `uv` yourself inside. |
| `run_python3_cpu.sh` | CPU Apptainer wrapper (no `#SBATCH --gpus`). |
| `run_python3_scavenger.sh` | Scavenger-partition variant (see file for `#SBATCH` lines). |
| `run_cell_eval.sh` / `run_cell_eval_imputed.sh` | Integration with [cell-eval](https://github.com/ArcInstitute/cell-eval); paths and module loads are site-specific. |
| `install_in_apptainer.sh` | Helper for installing inside an Apptainer image—read before use. |

## Typical submission pattern

```bash
# Edit config.yaml and any preprocess scripts for your paths first.
source setup_uv_env.sh
uv sync

sbatch run_train_state.sh config.yaml
# or
sbatch 01_run_train_state_config.sh config.yaml
```

`sbatch` uses `#SBATCH` directives at the top of each script. For local debugging, many scripts also work as `bash run_train_state.sh` from the repo root.

## Configuration

- **`config.yaml`**: Central YAML for this fork’s drivers (`run_train_state.sh`, inference, ISP). Schema is documented in [RUNNING_MODELS.md](../RUNNING_MODELS.md). **Do not commit secrets** (tokens, private wandb entity names if policy requires); use environment variables where the CLI supports them.
- **TOML task files** (e.g. `examples/fewshot.toml`) are unchanged from upstream conventions; paths inside them must exist on the cluster filesystem.

## Apptainer vs `uv run state`

- **`run_python3*.sh`**: General Python inside a fixed SIF (`helical_sb`). Binds include `/iridisfs/ddnb/Luke`—change for other users or clusters.
- **`uv run state ...`**: Uses the project’s `.venv` / `uv` resolution per `pyproject.toml`. Prefer this for STATE training, inference, and preprocessing unless you have a containerized stack that mirrors the same versions.

## Pushing this fork to GitHub

Recommended remote name: **`cxp_state`** (this checkout). Upstream Arc repo stays as `upstream`:

```bash
cd /path/to/cxp_state
git remote add cxp_state git@github.com:<your-username>/cxp_state.git   # once
git push -u cxp_state main
```

If you still use `origin` → `State-wrapper`, you can push the same branch there or repoint `origin` to `cxp_state`.
