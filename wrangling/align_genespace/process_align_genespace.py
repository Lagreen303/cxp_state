"""
Per-dataset preprocessing pipeline for STATE with shared gene space.

What it does:

1) Per-dataset:
   - Load .h5ad
   - Harmonise obs:
       * cytokine/perturbation/gene   -> gene
       * celltype/cell_line/cell_type -> cell_type
       * batch/sample                 -> batch
     + remap control label (e.g. PBS -> non-targeting)
   - Keep counts in X / layers["counts"] (CSR, float32, non-negative)

2) Across datasets:
   - Compute intersection of genes present in ALL datasets.
   - Use the reference dataset to define the canonical gene order.

3) Per-dataset again (independent preprocessing, no concatenation):
   - Subset to common genes in reference order.
   - Run:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=NUM_HVGS)
   - DO NOT subset the AnnData to HVGs.
   - Store HVG expression in:
        adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()
   - Save each dataset to OUTPUT_DIR as <stem>_state_preprocessed.h5ad.
"""

import os
from pathlib import Path

# =========================
# CONFIG — EDIT THIS BLOCK
# =========================

# Output directory for ALL processed datasets
OUTPUT_DIR = "/iridisfs/ddnb/Luke/data/replogle_cytokine_experiment"

# Reference dataset (defines preferred gene ordering)
REF_PATH = "/iridisfs/ddnb/Luke/data/Replogle/ReplogleWeissman2022_K562_essential.h5ad"

# Other datasets to process alongside the reference
OTHER_PATHS = [
    "/iridisfs/ddnb/Luke/data/Replogle/ReplogleWeissman2022_rpe1.h5ad",
    "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/Parse_10M_PBMC_cytokines.h5ad"# ...
]

# How many HVGs to use for STATE
NUM_HVGS = 2000

# Basename suffix for each per-dataset preprocessed file
OUT_SUFFIX = "_state.h5ad"

# Max missing genes to print per dataset (for diagnostics)
MAX_PRINT_MISSING = 50

# Harmonisation candidate names
PERT_COL_CANDIDATES = ["cytokine", "perturbation", "gene"]
CELLTYPE_COL_CANDIDATES = ["celltype", "cell_line", "cell_type"]
BATCH_COL_CANDIDATES = ["batch", "sample"]

# Unified column names
PERT_COL_UNIFIED = "pert_gene"
CELLTYPE_COL_UNIFIED = "cell_type"
BATCH_COL_UNIFIED = "batch"

# Control label mapping (priority order — first match wins)
CONTROL_LABEL_CANDIDATES = ["PBS", "non-targeting", "control"]
CONTROL_LABEL_RENAMED = "control"

# =========================
# END CONFIG
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: Numba cache
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True, mode=0o755)

import anndata as ad
import numpy as np
import scipy.sparse as sp
import scanpy as sc


# ----------------- helpers ----------------- #

def harmonise_obs_columns(adata: ad.AnnData, name: str = "") -> ad.AnnData:
    """Map variable column names to unified schema (gene, cell_type, batch) with priority order."""
    a = adata.copy()
    cols = set(a.obs.columns)

    def pick(cands, role):
        found = [c for c in cands if c in cols]
        if not found:
            raise KeyError(f"[{name}] Missing {role} column: tried {cands}")
        selected = found[0]
        if len(found) > 1:
            print(
                f"[INFO][{name}] Multiple {role} candidates found: {found}. "
                f"Selected '{selected}' (highest priority)",
                flush=True,
            )
        return selected

    pert_col = pick(PERT_COL_CANDIDATES, "perturbation")
    celltype_col = pick(CELLTYPE_COL_CANDIDATES, "cell_type")
    batch_col = pick(BATCH_COL_CANDIDATES, "batch")

    a.obs.rename(
        columns={
            pert_col: PERT_COL_UNIFIED,
            celltype_col: CELLTYPE_COL_UNIFIED,
            batch_col: BATCH_COL_UNIFIED,
        },
        inplace=True,
    )

    # Ensure key obs columns are categorical so h5ad stores obs/<col>/categories
    for col in (PERT_COL_UNIFIED, CELLTYPE_COL_UNIFIED, BATCH_COL_UNIFIED):
        if col in a.obs.columns:
            a.obs[col] = a.obs[col].astype("category")

    print(
        f"[INFO][{name}] Harmonised columns: "
        f"{pert_col}→{PERT_COL_UNIFIED}, "
        f"{celltype_col}→{CELLTYPE_COL_UNIFIED}, "
        f"{batch_col}→{BATCH_COL_UNIFIED}",
        flush=True,
    )
    return a


def harmonise_one(adata: ad.AnnData, name: str) -> ad.AnnData:
    """Full per-dataset harmonisation including control relabel."""
    print(f"[INFO][{name}] Harmonising dataset...", flush=True)
    a = harmonise_obs_columns(adata, name=name)

    # control relabel with hierarchy
    pert_series = a.obs[PERT_COL_UNIFIED].astype(str)
    unique_labels = set(pert_series.unique())
    control_found = None
    for candidate in CONTROL_LABEL_CANDIDATES:
        if candidate in unique_labels:
            control_found = candidate
            break

    if control_found is not None:
        pert_series = pert_series.replace({control_found: CONTROL_LABEL_RENAMED})
        print(
            f"[INFO][{name}] Control '{control_found}'→'{CONTROL_LABEL_RENAMED}'",
            flush=True,
        )
    else:
        print(
            f"[WARN][{name}] No control label found among {CONTROL_LABEL_CANDIDATES}",
            flush=True,
        )

    a.obs[PERT_COL_UNIFIED] = pert_series.astype("category")

    # ensure counts in X
    if "counts" in a.layers:
        counts = a.layers["counts"]
        print(f"[INFO][{name}] Using layers['counts'] as counts", flush=True)
    else:
        counts = a.X
        print(f"[INFO][{name}] Using X as counts", flush=True)

    if sp.issparse(counts):
        counts = counts.tocsr().astype(np.float32, copy=False)
        counts.data[counts.data < 0] = 0
    else:
        counts = np.asarray(counts, dtype=np.float32)
        counts[counts < 0] = 0
        counts = sp.csr_matrix(counts)

    a.X = counts
    a.layers["counts"] = counts

    print(
        f"[INFO][{name}] Harmonised shape: {a.n_obs:,} cells × {a.n_vars:,} genes",
        flush=True,
    )
    return a


def preprocess_one_to_common_genes(
    adata: ad.AnnData,
    name: str,
    common_order: list[str],
    out_dir: Path,
):
    """
    For a single dataset:
      - subset to common genes in common_order
      - normalize_total, log1p, highly_variable_genes
      - store HVG expression in .obsm["X_hvg"]
      - save to OUTPUT_DIR
    """
    print(f"[INFO][{name}] Subsetting to common genes (n={len(common_order):,})", flush=True)
    missing = [g for g in common_order if g not in adata.var_names]
    if missing:
        # Should not happen if common_order is computed correctly
        raise RuntimeError(f"[ERROR][{name}] Missing {len(missing)} 'common' genes.")

    a = adata[:, common_order].copy()
    print(f"[INFO][{name}] After subsetting: {a.n_obs:,} cells × {a.n_vars:,} genes", flush=True)

    # Ensure X is still CSR float32, non-negative
    X = a.layers.get("counts", a.X)
    if sp.issparse(X):
        X = X.tocsr().astype(np.float32, copy=False)
        X.data[X.data < 0] = 0
    else:
        X = np.asarray(X, dtype=np.float32)
        X[X < 0] = 0
        X = sp.csr_matrix(X)

    a.X = X
    a.layers["counts"] = X

    # STATE-style per-dataset preprocessing
    print(f"[INFO][{name}] Normalizing total counts per cell...", flush=True)
    sc.pp.normalize_total(a)

    print(f"[INFO][{name}] Applying log1p...", flush=True)
    sc.pp.log1p(a)

    print(f"[INFO][{name}] Finding top {NUM_HVGS} highly variable genes...", flush=True)
    sc.pp.highly_variable_genes(a, n_top_genes=NUM_HVGS)

    if "highly_variable" not in a.var.columns:
        raise RuntimeError(f"[ERROR][{name}] 'highly_variable' column missing after HVG computation.")

    n_hvg = int(a.var["highly_variable"].sum())
    print(f"[INFO][{name}] HVGs selected: {n_hvg:,}", flush=True)

    # DO NOT subset the AnnData to HVGs; just store HVG expression
    hvg_mask = a.var["highly_variable"].values
    print(f"[INFO][{name}] Storing HVG expression in obsm['X_hvg']...", flush=True)
    a.obsm["X_hvg"] = a[:, hvg_mask].X.toarray()

    # Save
    out_path = out_dir / f"{name}{OUT_SUFFIX}"
    print(f"[INFO][{name}] Writing preprocessed AnnData to: {out_path}", flush=True)
    a.write_h5ad(out_path, compression="gzip")
    print(f"[OK][{name}] Done.", flush=True)


# ----------------- main ----------------- #

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load + harmonise all datasets ----------
    ref_path = Path(REF_PATH)
    print(f"[INFO] Loading reference: {ref_path}", flush=True)
    ref_raw = ad.read_h5ad(ref_path)
    ref = harmonise_one(ref_raw, ref_path.stem)

    datasets = [(ref_path.stem, ref)]

    for p in OTHER_PATHS:
        path = Path(p)
        print("\n" + "=" * 80, flush=True)
        print(f"[INFO] Loading dataset: {path}", flush=True)
        a_raw = ad.read_h5ad(path)
        a = harmonise_one(a_raw, path.stem)
        datasets.append((path.stem, a))

    # ---------- 2. Compute genes present in ALL datasets ----------
    print("\n" + "=" * 80, flush=True)
    print("[INFO] Computing intersection of genes across ALL datasets...", flush=True)

    gene_sets = []
    for name, a in datasets:
        genes = set(a.var_names)
        gene_sets.append(genes)
        print(f"[INFO][{name}] genes: {len(genes):,}", flush=True)

    common_genes = set.intersection(*gene_sets)
    print(f"[INFO] Genes present in ALL datasets: {len(common_genes):,}", flush=True)

    if len(common_genes) == 0:
        raise RuntimeError("[ERROR] No common genes across all datasets.")

    # reference-based ordering for the common genes
    ref_genes = ref.var_names.tolist()
    common_order = [g for g in ref_genes if g in common_genes]
    print(f"[INFO] Common genes in reference order: {len(common_order):,}", flush=True)

    # ---------- 3. Per-dataset independent preprocessing ----------
    print("\n" + "=" * 80, flush=True)
    print("[INFO] Preprocessing each dataset independently (no concatenation)...", flush=True)

    for name, a in datasets:
        print("\n" + "-" * 60, flush=True)
        preprocess_one_to_common_genes(a, name, common_order, out_dir)

    print("\n[DONE] All datasets harmonised, restricted to common genes, "
          "preprocessed independently, and saved.", flush=True)


if __name__ == "__main__":
    main()
