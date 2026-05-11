#!/usr/bin/env python3
"""
Generate 9 cytokine fold TOMLs for 9-fold CV.

- Reads the base cytokine.toml to get:
  - dataset path (datasets.cytokine)
  - existing [fewshot] cell-type keys

- Reads the .h5ad, gets unique cytokines from obs['cytokine'],
  drops PBS, shuffles, and splits into 9 folds of 10.

- For each fold K, writes a TOML:

    cross_validation/tomls/cytokine_foldK.toml

  where for every [fewshot."cytokine.<celltype>"]:

    val  = [10 cytokines for fold K]
    test = [same 10 cytokines]

so training still uses val for validation, and `state tx predict` has
a non-empty test set for evaluation.
"""

import json
from copy import deepcopy
from pathlib import Path

import anndata as ad
import numpy as np
import toml  # pip install toml


# --- Paths / constants -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # .../cytokine_validation_2026
BASE_TOML = BASE_DIR / "cytokine.toml"

CV_DIR = BASE_DIR / "cross_validation"
TOML_DIR = CV_DIR / "tomls"

N_FOLDS = 9
FOLD_SIZE = 10

# Column in .h5ad containing cytokine names
PERT_COL = "cytokine"  # matches training.data.pert_col in config.yaml
CONTROL_PERT = "PBS"   # control to drop before splitting folds


# --- Helpers -----------------------------------------------------------------


def chunk_list(xs, n_chunks):
    """Split list xs into n_chunks chunks as evenly as possible, preserving order."""
    xs = list(xs)
    n = len(xs)
    base, extra = divmod(n, n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        end = start + size
        chunks.append(xs[start:end])
        start = end
    return chunks


# --- Main logic --------------------------------------------------------------


def main():
    # 1) Load base TOML template
    if not BASE_TOML.exists():
        raise FileNotFoundError(f"Base TOML not found: {BASE_TOML}")

    base_cfg = toml.load(BASE_TOML)

    # Ensure output dir for folds exists
    TOML_DIR.mkdir(parents=True, exist_ok=True)

    # Get dataset path from [datasets]
    datasets_cfg = base_cfg.get("datasets", {})
    cytokine_path = datasets_cfg.get("cytokine")
    if cytokine_path is None:
        raise KeyError("Missing [datasets].cytokine in base TOML")

    cytokine_path = Path(cytokine_path)

    # 2) Load .h5ad and extract cytokines
    if not cytokine_path.exists():
        raise FileNotFoundError(f".h5ad not found at {cytokine_path}")

    print(f"Loading AnnData from {cytokine_path} ...")
    adata = ad.read_h5ad(cytokine_path)

    if PERT_COL not in adata.obs:
        raise KeyError(f"Column '{PERT_COL}' not found in adata.obs")

    cytokines = sorted(map(str, adata.obs[PERT_COL].unique().tolist()))
    n_cyt_total = len(cytokines)
    print(f"Found {n_cyt_total} unique cytokines (including control, if present).")

    # Drop control cytokine (e.g. PBS) from the set before defining folds
    if CONTROL_PERT in cytokines:
        cytokines = [c for c in cytokines if c != CONTROL_PERT]
        print(f"Excluding control perturbation '{CONTROL_PERT}' from folds.")

    n_cyt = len(cytokines)
    print(f"Using {n_cyt} cytokines for cross-validation.")

    if n_cyt != N_FOLDS * FOLD_SIZE:
        raise ValueError(
            f"Expected {N_FOLDS * FOLD_SIZE} cytokines (excluding control), found {n_cyt}. "
            f"Adjust N_FOLDS / FOLD_SIZE or inspect the data."
        )

    # 3) Randomly assign cytokines to folds (with fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    cytokines_shuffled = cytokines.copy()
    rng.shuffle(cytokines_shuffled)

    folds = chunk_list(cytokines_shuffled, N_FOLDS)
    assert all(len(f) == FOLD_SIZE for f in folds), "Uneven fold sizes!"

    # 4) Discover existing fewshot cell-type keys from the base TOML
    fewshot_cfg = base_cfg.get("fewshot", {})
    if not fewshot_cfg:
        raise KeyError("Base TOML has no [fewshot] section to clone from.")

    celltype_keys = list(fewshot_cfg.keys())
    print(f"Using {len(celltype_keys)} cell-type fewshot keys from base TOML:")
    for k in celltype_keys:
        print(f"  - {k}")

    # 5) Write an optional JSON summary of fold assignments
    folds_json_path = BASE_DIR / "cytokine_folds.json"
    with folds_json_path.open("w") as f:
        json.dump(
            {
                "n_folds": N_FOLDS,
                "folds": {str(i + 1): folds[i] for i in range(N_FOLDS)},
            },
            f,
            indent=2,
        )
    print(f"Saved fold definition to {folds_json_path}")

    # 6) Generate one TOML per fold
    for fold_idx, fold_cyts in enumerate(folds, start=1):
        cfg_k = deepcopy(base_cfg)

        # Build new [fewshot] for this fold:
        #   same keys, and val/test both set to the fold's 10 cytokines.
        new_fewshot = {}
        for key in celltype_keys:
            new_fewshot[key] = {
                "val": fold_cyts,
                "test": fold_cyts,
            }

        cfg_k["fewshot"] = new_fewshot

        out_toml = TOML_DIR / f"cytokine_fold{fold_idx}.toml"
        with out_toml.open("w") as f:
            toml.dump(cfg_k, f)

        print(f"Wrote {out_toml} with {len(fold_cyts)} val/test cytokines.")

    print("Done. 9 TOML configs generated.")


if __name__ == "__main__":
    main()