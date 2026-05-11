# STATE — maintained fork (HPC, YAML workflow, experiment tooling)

This repository is a **personal fork** of the Arc Institute [**state**](https://github.com/ArcInstitute/state) project: the same `arc-state` Python package (`state` CLI: `emb`, `tx`) lives under `src/`, with **additional** Slurm/Apptainer helpers, YAML-driven training drivers, documentation for this workflow, and **version-controlled analysis / cross-validation (CV) code** under `experiments/`. It is **not** the official Arc Institute distribution; cite Arc’s work when you use their model or code, and keep their license files in any publication fork.

**Upstream reference:** [ArcInstitute/state](https://github.com/ArcInstitute/state) — paper: [State (bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2). Colabs linked on the upstream repo remain the quickest way to try STATE without a cluster.

---

## What this fork adds (beyond upstream `state`)

| Area | Location | Purpose |
|------|-----------|---------|
| **HPC / Slurm wrappers** | Repo root (`run_train_state.sh`, `run_python3*.sh`, `00_preprocess_train.sh`, `01_run_train_state_config.sh`, `02_run_inference_state_config.sh`, `02a_preprocess_infer.sh`, cell-eval drivers, …) | Submit or debug training, inference, preprocessing, and Apptainer-backed Python on Iridis-style partitions. |
| **Environment** | `setup_uv_env.sh` | Project-local `uv` cache and `.venv` paths (override with env vars if your site uses scratch). |
| **YAML + TOML workflow** | `RUNNING_MODELS.md`, optional `config.yaml` (local) | Describes how YAML maps to `state tx train` / infer / ISP for this fork’s drivers. |
| **Wrapper index** | `docs/HPC_SLURM_WRAPPERS.md` | One-page map of each shell script and how to repoint `git` remotes (`upstream` = Arc, `origin` = your GitHub). |
| **CV & analysis code** | `experiments/**` | Fold generation, aggregation, plotting scripts, Slurm snippets, and **small config** (`.yaml`, `.toml`, `.md`). **No** AnnData, checkpoints, or bulk results are tracked (see below). |
| **Standalone scripts** | Repo root | `classify_from_embedding.py`, `cluster_IFNy_PBMCs_cluster_h5ad.py`, `download_repogle_data.py` — ad hoc analysis / data helpers (paths are machine-specific; edit before use). |

Upstream-facing install and API details are still valid; this README focuses on **using this tree** on a cluster and **what is safe to push**.

---

## What is tracked under `experiments/` (and what is not)

Git is configured so **`experiments/`** can hold reproducible **code and configs** without committing large artifacts:

**Intended to be pushed**

- `*.py`, `*.sh`, `*.ipynb`
- `*.yaml`, `*.yml`, `*.toml`
- `*.md`, `*.txt`

**Ignored (not pushed)** — including but not limited to:

- `*.h5ad`, `*.h5`, Zarr/Loom, `*.mtx`, archives
- `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`, `*.npz`, `*.npy`, pickles
- `checkpoints/`, `wandb/`, `lightning_logs/`, `outputs/`, `plots/`
- `*.csv`, `*.tsv` (regenerate from the Python pipeline)
- Images and heavy binaries (`*.png`, `*.pdf`, `*.svg`, …) under `experiments/`

So: **push CV/analysis scripts and experiment configs; do not push data or run outputs.** If you need a tiny example table in git, keep it under `docs/` or `examples/` instead of `experiments/`.

---

## Installation (from this fork)

```bash
git clone <your-github-url>/state.git   # or your fork name
cd state
source setup_uv_env.sh                  # optional but recommended on shared systems
uv sync
uv run state --help
```

Editable install (when changing `src/`):

```bash
uv tool install -e .
```

PyPI package name remains **`arc-state`** for the upstream line; this fork may carry extra commits under the same package layout.

---

## Documentation map

| Doc | Use |
|-----|-----|
| **[docs/HPC_SLURM_WRAPPERS.md](docs/HPC_SLURM_WRAPPERS.md)** | Which Slurm / Apptainer script to run and how it relates to `uv run state`. |
| **[RUNNING_MODELS.md](RUNNING_MODELS.md)** | YAML + TOML workflow (preprocess → train → infer / ISP). |
| **`examples/`** | Upstream TOML examples (`fewshot.toml`, `zeroshot.toml`, …). |

---

## CLI overview (unchanged from upstream)

```bash
state --help          # emb | tx
state tx --help       # train, predict, infer, preprocess_*, …
state emb --help      # embedding model
```

**Minimal ST training shape** (Hydra overrides; paths and columns must match your AnnData):

```bash
uv run state tx train \
  data.kwargs.toml_config_path="examples/fewshot.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=TARGET1 \
  training.max_steps=40000 \
  model=pertsets \
  output_dir="$PWD/runs" \
  name="my_run"
```

**Preprocessing** (store HVG matrix in `.obsm['X_hvg']`):

```bash
uv run state tx preprocess_train \
  --adata /path/to/raw.h5ad \
  --output /path/to/processed.h5ad \
  --num_hvgs 2000
```

```bash
uv run state tx preprocess_infer \
  --adata /path/to/processed.h5ad \
  --output /path/to/control_template.h5ad \
  --control-condition "DMSO" \
  --pert-col "treatment" \
  --seed 42
```

For full TOML semantics (zeroshot / fewshot), long embedding examples, Singularity build, and vector DB extras, see the **[upstream README](https://github.com/ArcInstitute/state/blob/main/README.md)** and Arc’s Colabs.

---

## Related Arc repositories

- [cell-eval](https://github.com/ArcInstitute/cell-eval) — evaluation framework  
- [cell-load](https://github.com/ArcInstitute/cell-load) — dataloaders / preprocessing  

---

## Licenses

State **code** in this lineage is under [LICENSE](LICENSE) (CC BY-NC-SA 4.0). **Model weights** and use constraints are in [MODEL_LICENSE.md](MODEL_LICENSE.md) and [MODEL_ACCEPTABLE_USE_POLICY.md](MODEL_ACCEPTABLE_USE_POLICY.md).

Publications should cite the State [paper](https://arcinstitute.org/manuscripts/State) and clearly state whether results use the official Arc release or this fork.

---

## Google Colabs (upstream quickstarts)

- [Replogle-Nadig ST training](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN)  
- [Tahoe-100M inference](https://colab.research.google.com/drive/1bq5v7hixnM-tZHwNdgPiuuDo6kuiwLKJ)  
- [SE embed / annotate](https://colab.research.google.com/drive/1uJinTJLSesJeot0mP254fQpSxGuDEsZt)  
- [Virtual Cell Challenge](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l)  
