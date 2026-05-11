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

## Config YAML and the Slurm wrappers

The root Slurm drivers are **not** Hydra configs by themselves: each script expects a **YAML file** (usually the same style as in [RUNNING_MODELS.md](RUNNING_MODELS.md)) that documents your TOML paths, training hyperparameters, and/or inference paths. You maintain **one YAML per logical experiment** (or reuse a template under `experiments/.../configs/`).

**Passing the file**

- Most scripts accept an optional path: `sbatch run_train_state.sh path/to/config.yaml`. If omitted, they default to **`config.yaml` in the repo root** (when run from the submit directory, Slurm sets `SLURM_SUBMIT_DIR` so paths resolve next to the wrapper).
- Use **absolute paths inside YAML** when jobs start from another working directory, or keep everything relative to the directory you `sbatch` from.

**Which YAML section each wrapper reads**

| Wrapper | YAML section used | Result |
|--------|---------------------|--------|
| `run_train_state.sh`, `01_run_train_state_config.sh` | `training` (nested `data`, `training`, `wandb`, `output`, plus top-level `model`, optional `init_from`) | Builds `uv run state tx train ...` Hydra overrides. |
| `02_run_inference_state_config.sh`, `run_insilicoperturbation_state.sh` | `inference` | Builds `uv run state tx infer ...`. Required keys include `output`, `model_dir`, `checkpoint`, `adata`, `pert_col`; optional `control_pert`. If `checkpoint` is a bare filename, it is resolved under `<model_dir>/checkpoints/`. |
| `run_cell_eval.sh` (and imputed variant) | Same file must contain parseable **`inference.output`** (predictions `.h5ad`) and **`inference.real`** (reference `.h5ad`), plus `pert_col` and `control_pert` | See below. |

**Before any GPU job**

```bash
cd /path/to/this/repo
source setup_uv_env.sh   # project-local uv cache + .venv
uv sync
```

Preprocess helpers **`00_preprocess_train.sh`** and **`02a_preprocess_infer.sh`** are **path-hardcoded Slurm examples**, not YAML-driven; copy them and edit `uv run state tx preprocess_*` arguments.

---

## Wrapper scripts (what each main file does)

### Training

- **`run_train_state.sh`** — Primary training driver: sources `setup_uv_env.sh`, loads the given YAML, uses embedded Python to assemble a full **`uv run state tx train`** command (data kwargs, training schedule, `model=...`, optional **`init_from`**, wandb toggles/tags). Optional: `SKIP_STEPS=01` to skip the training step. Heavier and more flexible than `01_...`.
- **`01_run_train_state_config.sh`** — Shorter Slurm job that parses the same `training` YAML and runs `uv run state tx train` with a smaller fixed set of overrides (good when you do not need `init_from` or extra wandb handling). Usage: `sbatch 01_run_train_state_config.sh [config.yaml]`.

### Inference / in-silico perturbation

- **`run_insilicoperturbation_state.sh`** — Reads **`inference`** from YAML and runs **`state tx infer`** with resolved checkpoint path. Usage: `sbatch run_insilicoperturbation_state.sh [config.yaml]`; optional `SKIP_STEPS=02`.
- **`02_run_inference_state_config.sh`** — Same overall job (YAML → `state tx infer`) with a separate embedded parser; use whichever you prefer to maintain. Usage: `sbatch 02_run_inference_state_config.sh [config.yaml]`.
- **`run_predict_state.sh`** — **Template only**: a single hardcoded `uv run state tx predict ...` with fixed `--output-dir` and `--checkpoint`. Duplicate the file or edit in place for your run directory and checkpoint name.

### cell-eval style metrics (after you have prediction + real h5ads)

- **`run_cell_eval.sh`** — Submits a job that **does not train**; it evaluates predictions like STATE’s internal `tx predict` checks using **cell-eval–backed Python** in this repo. **Usage:** `sbatch run_cell_eval.sh /path/to/config_infer_….yaml` (exactly **one** argument). The script naïvely parses the YAML with `awk`: it needs **`inference`** fields **`output:`** (path to **predicted** `.h5ad`, quoted), **`real:`** (path to **real / observed** `.h5ad`, quoted), **`pert_col`**, and **`control_pert`**. It derives a sibling folder under the run directory, e.g. `results_eval_<prediction_basename>_genes_all`, and runs  
  `uv run … python analyse_results/state_predict_like_metrics.py` with `--profile full` and `cell_type` as the cell-type column. **Prepare your inference YAML** so those keys match the files produced by `state tx infer` / `predict`.

- **`run_cell_eval_imputed.sh`** — Same CLI contract (`CONFIG_YAML` as sole argument); calls `state_predict_like_metrics_no_imputed.py` so metrics exclude imputed genes (`--imputed-flag`, defaulting to `is_imputed`).

- **`run_insilicoperturbation_per_pert.sh`** — Batch ISP driver: YAML supplies **`templates_dir`** and **`predictions_dir`** (directory of template `.h5ad` files → many outputs).

### Apptainer-backed Python (not `config.yaml`-driven)

These run **`python3 "$@"`** inside a **site-specific** Apptainer image and bind mounts (`run_python3.sh` is GPU-oriented; `run_python3_cpu.sh` omits GPU in the header; `run_python3_scavenger.sh` targets scavenger partitions). They do **not** read your STATE YAML; use them for arbitrary Python in the same filesystem environment as your other jobs. Example: `sbatch run_python3.sh your_script.py --flags`.

---

## State embedding (SE) — where the “embedding scripts” are

There is **no** separate `scripts/state_embed_anndata.py` in this repository anymore; embedding is handled by the **upstream CLI** under **`state emb`** (implementation in `src/state/_cli/_emb/`: `preprocess`, `fit`, `transform`, `query`, etc.). Start with:

```bash
uv run state emb --help
uv run state emb transform --help
```

**Slurm / shell examples checked into this fork** (paths and partitions are for the original Iridis layout — edit before reuse):

- **`run_emb_transform.sh`** — YAML-driven **`state emb transform`** (repo root; default `emb_transform.yaml`, template **`examples/emb_transform.yaml`**).
- **`experiments/cytokine_validation_2026_03/emb_finetune/run_emb_finetune_cytokine.sh`** — `state emb preprocess` and `state emb fit` with a YAML config path.
- **`experiments/cytokine_validation_zeroshot_2026_04_01/emb_zeroshot/run_emb_transform_cytokine.sh`** — runs **`state emb transform`** over the cytokine atlas embedding workflow used in that campaign.

For notebook-style tutorials, use Arc’s **SE Colab** linked in [Google Colabs](#google-colabs-upstream-quickstarts) below.

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
