#!/usr/bin/env python3
"""
Build multiple template AnnDatas for STATE inference (CONFIG-ONLY).

For each cytokine label C in KEEP_PERTS:
- Subset input AnnData to:
    cell_type == CELL_TYPE_TO_KEEP AND INPUT_PERT_COL == C
- Optionally subset genes to COMMON_GENES_PATH
- Create TEST_PERT_COL such that:
    - A fraction of cells (CONTROL_FRACTION) keep TEST_PERT_COL = C  (controls)
    - Remaining cells get Replogle test perturbations cycled across cells
- Save one template per cytokine into OUTPUT_DIR

This is metadata wrangling intended for use BEFORE `state tx preprocess_inference`.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# ============================================================
# CONFIG — EDIT THIS BLOCK
# ============================================================

INPUT_H5AD = "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/Parse_10M_PBMC_cytokines_CD14Mono.h5ad"

OUTPUT_DIR = "/iridisfs/ddnb/Luke/scFMs/state/experiments/replogle_cytokine_experiment/templates_top10"

# Optional: common genes text file (one gene per line). Set to None to disable.
COMMON_GENES_PATH = "/iridisfs/ddnb/Luke/data/replogle_cytokine_experiment/common_genes.txt"
# COMMON_GENES_PATH = None

# Input obs columns
INPUT_CELLTYPE_COL = "cell_type"
INPUT_PERT_COL     = "cytokine"

CELL_TYPE_TO_KEEP = "CD14 Mono"

# Build one template per cytokine label in this list
KEEP_PERTS = [
    "IL-4",
    "M-CSF",
    "IL-13",
    "IL-32-beta",
    "IL-10",
    "IFN-omega",
    "GM-CSF",
    "IFN-beta",
    "IFN-gamma",
    "IL-1-beta",
]

# Fraction of cytokine-C cells to keep as "controls" => TEST_PERT_COL = C
# Remaining cells will be assigned Replogle test perturbations.
CONTROL_FRACTION = 0.10   # 10% controls, 90% assigned test perts

# New column name for test perturbations
TEST_PERT_COL = "pert_gene_test"

# Optional marker column showing which rows are controls
MARK_CONTROL_COL = "is_control_like"  # set to None to skip

# Perturbation dictionary AnnDatas (Replogle screens etc.)
PERT_DICT_PATHS = [
    "/iridisfs/ddnb/Luke/data/Replogle/ReplogleWeissman2022_K562_essential.h5ad",
    "/iridisfs/ddnb/Luke/data/Replogle/ReplogleWeissman2022_rpe1.h5ad",
]

# Column name in dictionary AnnDatas containing perturbation labels (genes)
DICT_PERT_COL = "gene"

# Pert labels to exclude from dictionary perturbations
EXCLUDE_PERTS = {"control", "non-targeting", "NT", "ntc", "PBS", "IFN-gamma"}

# Randomness (for selecting which cells are controls + shuffling perts)
SEED = 42
SHUFFLE_TEST_PERTS = True

# Output naming
OUT_PREFIX = "cd14mono"
OUT_SUFFIX = "__template_raw.h5ad"

# Sanitise output filenames
SAFE_REPLACE = {" ": "_", "/": "_", "\\": "_"}

# ============================================================
# END CONFIG
# ============================================================


def safe_name(s: str) -> str:
    out = s
    for k, v in SAFE_REPLACE.items():
        out = out.replace(k, v)
    return out


def load_common_genes(path: str | None) -> list[str] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] common genes file not found: {p}")
    genes = [g.strip() for g in p.read_text().splitlines() if g.strip()]
    if not genes:
        raise ValueError("[ERROR] common genes file is empty.")
    print(f"[INFO] Loaded {len(genes):,} common genes from {p}", flush=True)
    return genes


def subset_to_genes(adata: ad.AnnData, genes: list[str] | None, label: str) -> ad.AnnData:
    if genes is None:
        return adata
    present = [g for g in genes if g in adata.var_names]
    missing_n = len(genes) - len(present)
    print(
        f"[INFO] [{label}] Found {len(present):,} / {len(genes):,} common genes "
        f"({missing_n:,} missing)",
        flush=True,
    )
    if len(present) == 0:
        raise RuntimeError(f"[ERROR] After common-gene filtering, 0 genes remain for {label}.")
    return adata[:, present].copy()


def build_test_perturbation_list() -> list[str]:
    all_perts = []
    for p in PERT_DICT_PATHS:
        path = Path(p)
        print(f"[INFO] Loading perturbation dictionary: {path}", flush=True)
        adict = ad.read_h5ad(path)
        if DICT_PERT_COL not in adict.obs.columns:
            raise KeyError(f"[ERROR] '{DICT_PERT_COL}' not found in obs of {path}")
        all_perts.append(adict.obs[DICT_PERT_COL].astype(str))

    perts = pd.concat(all_perts, ignore_index=True)
    perts = perts[~perts.isna()]
    perts = perts[~perts.isin(EXCLUDE_PERTS)]
    unique = perts.drop_duplicates().tolist()

    if not unique:
        raise RuntimeError("[ERROR] No valid test perturbations after filtering.")

    if SHUFFLE_TEST_PERTS:
        rng = np.random.default_rng(SEED)
        rng.shuffle(unique)

    print(f"[INFO] Unique test perts after filtering: {len(unique):,}", flush=True)
    return unique


def assign_test_perts_within_cytokine_template(
    adata: ad.AnnData,
    cytokine_label: str,
    test_perts: list[str],
    control_fraction: float,
    seed: int,
) -> None:
    """
    Within a template where all cells are cytokine_label:
    - Select a fraction as controls => TEST_PERT_COL = cytokine_label
    - Assign remaining cells => TEST_PERT_COL = cycled test perts
    """
    if not (0.0 < control_fraction < 1.0):
        raise ValueError("[ERROR] CONTROL_FRACTION must be between 0 and 1 (exclusive).")

    n = adata.n_obs
    if n == 0:
        raise RuntimeError("[ERROR] Template has 0 cells after subsetting.")

    rng = np.random.default_rng(seed)

    # At least 1 control, at least 1 test row if possible
    n_control = int(round(n * control_fraction))
    n_control = max(1, min(n_control, n - 1)) if n > 1 else 1

    control_idx = rng.choice(n, size=n_control, replace=False)

    mask_control = np.zeros(n, dtype=bool)
    mask_control[control_idx] = True

    values = np.empty(n, dtype=object)
    values[mask_control] = cytokine_label

    n_non = n - n_control
    if n_non > 0:
        repeated = (test_perts * ((n_non // len(test_perts)) + 1))[:n_non]
        values[~mask_control] = repeated

    adata.obs[TEST_PERT_COL] = values

    if MARK_CONTROL_COL is not None:
        adata.obs[MARK_CONTROL_COL] = mask_control

    print(
        f"[INFO] Assigned {n_control:,} controls (test='{cytokine_label}') and "
        f"{n_non:,} test-pert rows (Replogle genes).",
        flush=True,
    )


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_genes = load_common_genes(COMMON_GENES_PATH)

    print(f"[INFO] Loading input AnnData: {INPUT_H5AD}", flush=True)
    adata_full = ad.read_h5ad(INPUT_H5AD)

    # Optional gene filtering once
    adata_full = subset_to_genes(adata_full, common_genes, label="input")

    for col in (INPUT_CELLTYPE_COL, INPUT_PERT_COL):
        if col not in adata_full.obs.columns:
            raise KeyError(f"[ERROR] Missing obs column '{col}' in input AnnData.")

    # Subset to cell type once
    mask_ct = adata_full.obs[INPUT_CELLTYPE_COL].astype(str).values == CELL_TYPE_TO_KEEP
    n_ct = int(mask_ct.sum())
    print(f"[INFO] {CELL_TYPE_TO_KEEP} cells in input: {n_ct:,}", flush=True)
    if n_ct == 0:
        raise RuntimeError(f"[ERROR] No cells for CELL_TYPE_TO_KEEP='{CELL_TYPE_TO_KEEP}'")

    adata_ct = adata_full[mask_ct].copy()
    print(
        f"[INFO] Cell-type subset shape: {adata_ct.n_obs:,} cells × {adata_ct.n_vars:,} genes",
        flush=True,
    )

    # Build test perturbations once
    test_perts = build_test_perturbation_list()

    available = set(adata_ct.obs[INPUT_PERT_COL].astype(str).unique())
    built = 0

    for cytok in KEEP_PERTS:
        print("\n" + "=" * 80, flush=True)

        if cytok not in available:
            print(f"[WARN] '{cytok}' not found in input. Skipping.", flush=True)
            continue

        mask_p = adata_ct.obs[INPUT_PERT_COL].astype(str).values == cytok
        n = int(mask_p.sum())
        print(f"[INFO] Cytokine='{cytok}' cells: {n:,}", flush=True)

        tmp = adata_ct[mask_p].copy()

        # Assign within-cytokine control/test split
        assign_test_perts_within_cytokine_template(
            tmp,
            cytokine_label=cytok,
            test_perts=test_perts,
            control_fraction=CONTROL_FRACTION,
            seed=SEED,
        )

        preview_cols = [INPUT_PERT_COL, TEST_PERT_COL]
        if MARK_CONTROL_COL is not None:
            preview_cols.append(MARK_CONTROL_COL)
        print(f"[INFO] Preview obs[{preview_cols}]:", flush=True)
        print(tmp.obs[preview_cols].head(), flush=True)

        out_name = f"{safe_name(OUT_PREFIX)}__{safe_name(cytok)}{OUT_SUFFIX}"
        out_path = out_dir / out_name
        print(f"[INFO] Writing template → {out_path}", flush=True)
        tmp.write_h5ad(out_path, compression="gzip")

        built += 1

    print("\n" + "=" * 80, flush=True)
    print(f"[OK] Done. Built {built} template(s) in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
