#!/usr/bin/env python3
"""
Per-dataset preprocessing pipeline for STATE with shared gene space.

Goal (your latest requirement):
  - Process the REFERENCE dataset first:
      * apply per-dataset gene naming
      * harmonise obs + counts
      * subset to COMMON genes (in reference ordering)
      * run: normalize_total -> log1p -> highly_variable_genes(n_top_genes=NUM_HVGS)
      * extract the HVG gene list (from within COMMON genes)
  - Process OTHER dataset(s):
      * apply per-dataset gene naming
      * harmonise obs + counts
      * subset to the SAME HVG GENE LIST chosen from the reference (and in that order)
      * run: normalize_total -> log1p
      * store expression for that SAME HVG list in obsm["X_hvg"] (dense)
  - Save <stem>_aligned.h5ad for each dataset

This guarantees ALL datasets share the same 2000-gene feature space, aligned to the
reference-selected HVGs (chosen only among common genes).
"""

import os
from pathlib import Path

# =========================
# CONFIG — EDIT THIS BLOCK
# =========================

OUTPUT_DIR = "/iridisfs/ddnb/Luke/data/cytokine_1k1k"

REF_PATH = "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/Parse_10M_PBMC_cytokines.h5ad"

OTHER_PATHS = [
    "/iridisfs/ddnb/Luke/data/Onek1k/2a930352-0802-4af7-bd26-61f75c83e1b0/eQTLAutoimmune.h5ad",
]

COMMON_GENE_FILE = "/iridisfs/ddnb/Luke/data/Onek1k/cytokine_1k1k_parse_common_genes.txt"

NUM_HVGS = 2000
OUT_SUFFIX = "_aligned.h5ad"
MAX_PRINT_MISSING = 50

PERT_COL_CANDIDATES = ["cytokine", "perturbation", "gene"]
CELLTYPE_COL_CANDIDATES = ["celltype", "cell_line", "cell_type"]
BATCH_COL_CANDIDATES = ["batch", "sample"]

PERT_COL_UNIFIED = "cytokine"
CELLTYPE_COL_UNIFIED = "cell_type"
BATCH_COL_UNIFIED = "batch"

CONTROL_LABEL_CANDIDATES = ["PBS", "non-targeting", "control"]
CONTROL_LABEL_RENAMED = "PBS"

# Per-dataset gene-name source:
#   - None => use adata.var_names
#   - "feature_name" (or any var column) => use adata.var["feature_name"]
GENE_NAME_SOURCE = {
    REF_PATH: None,  # set to "feature_name" etc. if needed
    "/iridisfs/ddnb/Luke/data/Onek1k/2a930352-0802-4af7-bd26-61f75c83e1b0/eQTLAutoimmune.h5ad": "feature_name",
}

DEFAULT_GENE_SOURCE_IF_MISSING = "error"
MAKE_GENE_NAMES_UNIQUE = True

# =========================
# END CONFIG
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True, mode=0o755)

import anndata as ad
import numpy as np
import scipy.sparse as sp
import scanpy as sc


# ----------------- gene naming helpers ----------------- #

def _make_unique(names: list[str]) -> list[str]:
    """Make names unique by appending __dupN deterministically."""
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}__dup{seen[n]}")
    return out


def _resolve_gene_source(dataset_path: str) -> str | None:
    if dataset_path in GENE_NAME_SOURCE:
        return GENE_NAME_SOURCE[dataset_path]

    if DEFAULT_GENE_SOURCE_IF_MISSING == "error":
        raise KeyError(
            f"[ERROR] No gene-name source configured for dataset:\n  {dataset_path}\n"
            "Add it to GENE_NAME_SOURCE or set DEFAULT_GENE_SOURCE_IF_MISSING."
        )

    if DEFAULT_GENE_SOURCE_IF_MISSING == "var_names":
        return None

    raise ValueError(
        "[ERROR] DEFAULT_GENE_SOURCE_IF_MISSING must be one of: 'error', 'var_names'"
    )


def apply_gene_names(adata: ad.AnnData, dataset_path: str, name: str) -> ad.AnnData:
    """
    Ensure adata.var_names contains gene symbols for this dataset.
    If configured source is a var column (e.g. feature_name), replace var_names with that.
    """
    gene_col = _resolve_gene_source(dataset_path)

    if gene_col is None:
        adata.var_names = adata.var_names.astype(str)
        return adata

    if gene_col not in adata.var.columns:
        raise KeyError(
            f"[ERROR][{name}] Gene column '{gene_col}' not found in adata.var.\n"
            f"Available var columns: {list(adata.var.columns)}"
        )

    new_names = adata.var[gene_col].astype(str).tolist()
    if len(new_names) != adata.n_vars:
        raise RuntimeError(
            f"[ERROR][{name}] var['{gene_col}'] length ({len(new_names)}) != n_vars ({adata.n_vars})"
        )

    if "_orig_var_names" not in adata.var.columns:
        adata.var["_orig_var_names"] = adata.var_names.astype(str).values

    bad = [i for i, g in enumerate(new_names) if (g is None) or (str(g).strip() == "") or (str(g).lower() == "nan")]
    if bad:
        raise RuntimeError(f"[ERROR][{name}] Found {len(bad)} empty/NaN gene names in var['{gene_col}'].")

    if len(set(new_names)) != len(new_names):
        n_dup = len(new_names) - len(set(new_names))
        if not MAKE_GENE_NAMES_UNIQUE:
            raise RuntimeError(
                f"[ERROR][{name}] Gene names from var['{gene_col}'] have duplicates (dup_count={n_dup}). "
                "Set MAKE_GENE_NAMES_UNIQUE=True or fix upstream."
            )
        print(
            f"[WARN][{name}] Gene names from var['{gene_col}'] have duplicates (dup_count={n_dup}); "
            "making unique with __dupN suffix.",
            flush=True,
        )
        new_names = _make_unique(new_names)

    adata.var_names = np.array(new_names, dtype=object)
    return adata


# ----------------- harmonisation helpers ----------------- #

def _pick_optional(cands: list[str], cols: set, role: str, name: str) -> str | None:
    """Return first matching column name, or None if none found (optional harmonisation)."""
    found = [c for c in cands if c in cols]
    if not found:
        return None
    selected = found[0]
    if len(found) > 1:
        print(
            f"[INFO][{name}] Multiple {role} candidates found: {found}. "
            f"Selected '{selected}' (highest priority)",
            flush=True,
        )
    return selected


def harmonise_obs_columns(adata: ad.AnnData, name: str = "") -> ad.AnnData:
    """
    Map variable column names to unified schema (cytokine, cell_type, batch) with priority order.
    If a column is not found (e.g. no perturbation in scRNA-seq), a default column is added
    so downstream subsetting/processing can still run.
    """
    a = adata.copy()
    cols = set(a.obs.columns)

    pert_col = _pick_optional(PERT_COL_CANDIDATES, cols, "perturbation", name)
    celltype_col = _pick_optional(CELLTYPE_COL_CANDIDATES, cols, "cell_type", name)
    batch_col = _pick_optional(BATCH_COL_CANDIDATES, cols, "batch", name)

    rename_map = {}
    if pert_col is not None:
        rename_map[pert_col] = PERT_COL_UNIFIED
    if celltype_col is not None:
        rename_map[celltype_col] = CELLTYPE_COL_UNIFIED
    if batch_col is not None:
        rename_map[batch_col] = BATCH_COL_UNIFIED

    if rename_map:
        a.obs.rename(columns=rename_map, inplace=True)

    # Add default columns for any missing (e.g. scRNA-seq without perturbation)
    if PERT_COL_UNIFIED not in a.obs.columns:
        a.obs[PERT_COL_UNIFIED] = "unperturbed"
        print(f"[INFO][{name}] No perturbation column found; added '{PERT_COL_UNIFIED}'='unperturbed'", flush=True)
    if CELLTYPE_COL_UNIFIED not in a.obs.columns:
        a.obs[CELLTYPE_COL_UNIFIED] = "unknown"
        print(f"[INFO][{name}] No cell_type column found; added '{CELLTYPE_COL_UNIFIED}'='unknown'", flush=True)
    if BATCH_COL_UNIFIED not in a.obs.columns:
        a.obs[BATCH_COL_UNIFIED] = name or "batch_0"
        print(f"[INFO][{name}] No batch column found; added '{BATCH_COL_UNIFIED}'='{name or \"batch_0\"}'", flush=True)

    for col in (PERT_COL_UNIFIED, CELLTYPE_COL_UNIFIED, BATCH_COL_UNIFIED):
        if col in a.obs.columns:
            a.obs[col] = a.obs[col].astype("category")

    parts = [
        f"{pert_col or '(default)'}→{PERT_COL_UNIFIED}",
        f"{celltype_col or '(default)'}→{CELLTYPE_COL_UNIFIED}",
        f"{batch_col or '(default)'}→{BATCH_COL_UNIFIED}",
    ]
    print(f"[INFO][{name}] Harmonised columns: {', '.join(parts)}", flush=True)
    return a


def harmonise_one(adata: ad.AnnData, name: str) -> ad.AnnData:
    """Full per-dataset harmonisation including control relabel + enforcing counts in X/layers['counts']."""
    print(f"[INFO][{name}] Harmonising dataset...", flush=True)
    a = harmonise_obs_columns(adata, name=name)

    pert_series = a.obs[PERT_COL_UNIFIED].astype(str)
    unique_labels = set(pert_series.unique())
    control_found = None
    for candidate in CONTROL_LABEL_CANDIDATES:
        if candidate in unique_labels:
            control_found = candidate
            break

    if control_found is not None:
        pert_series = pert_series.replace({control_found: CONTROL_LABEL_RENAMED})
        print(f"[INFO][{name}] Control '{control_found}'→'{CONTROL_LABEL_RENAMED}'", flush=True)
    else:
        print(f"[WARN][{name}] No control label found among {CONTROL_LABEL_CANDIDATES}", flush=True)

    a.obs[PERT_COL_UNIFIED] = pert_series.astype("category")

    if "counts" in a.layers:
        counts = a.layers["counts"]
        print(f"[INFO][{name}] Using layers['counts'] as counts", flush=True)
    else:
        counts = a.X
        print(f"[INFO][{name}] Using X as counts", flush=True)

    if sp.issparse(counts):
        counts = counts.tocsr().astype(np.float32, copy=False)
        if counts.nnz > 0:
            counts.data[counts.data < 0] = 0
    else:
        counts = np.asarray(counts, dtype=np.float32)
        counts[counts < 0] = 0
        counts = sp.csr_matrix(counts)

    a.X = counts
    a.layers["counts"] = counts

    print(f"[INFO][{name}] Harmonised shape: {a.n_obs:,} cells × {a.n_vars:,} genes", flush=True)
    return a


# ----------------- reference HVG selection (within common genes) ----------------- #

def compute_ref_hvg_gene_list(common_genes: list[str]) -> list[str]:
    """
    Load reference, apply gene naming + harmonisation, subset to common genes in ref order,
    then run normalize_total/log1p/HVG and return the HVG gene list (ordered as in adata.var_names).
    If the reference output _aligned.h5ad already exists, load it and return the HVG list without reprocessing.
    """
    ref_name = Path(REF_PATH).stem
    ref_path_str = str(Path(REF_PATH))
    out_path = Path(OUTPUT_DIR) / f"{ref_name}{OUT_SUFFIX}"

    if out_path.exists():
        print("\n" + "=" * 80, flush=True)
        print(f"[INFO][REF:{ref_name}] Output already exists: {out_path} — loading HVG list and skipping.", flush=True)
        adata_existing = ad.read_h5ad(out_path)
        if "highly_variable" not in adata_existing.var.columns:
            raise RuntimeError(
                f"[ERROR][REF:{ref_name}] Existing file missing 'highly_variable'; delete it to recompute."
            )
        hvg_genes = adata_existing.var_names[adata_existing.var["highly_variable"].values].astype(str).tolist()
        print(f"[OK][REF:{ref_name}] Loaded {len(hvg_genes):,} HVG genes from existing file.", flush=True)
        del adata_existing
        return hvg_genes

    print("\n" + "=" * 80, flush=True)
    print(f"[INFO][REF:{ref_name}] Loading reference dataset: {REF_PATH}", flush=True)
    adata_raw = ad.read_h5ad(REF_PATH)

    adata_raw = apply_gene_names(adata_raw, dataset_path=ref_path_str, name=ref_name)
    adata = harmonise_one(adata_raw, name=ref_name)
    del adata_raw

    # reference order after gene naming is just adata.var_names order
    ref_gene_order = list(adata.var_names.astype(str))
    ordered_common = [g for g in ref_gene_order if g in set(common_genes)]

    if len(ordered_common) == 0:
        raise RuntimeError("[ERROR][REF] 0 common genes found in reference after naming logic.")

    print(f"[INFO][REF:{ref_name}] Subsetting reference to common genes: {len(ordered_common):,}", flush=True)
    adata = adata[:, ordered_common].copy()
    print(f"[INFO][REF:{ref_name}] After common subsetting: {adata.n_obs:,} × {adata.n_vars:,}", flush=True)

    # replicate snippet
    print(f"[INFO][REF:{ref_name}] Normalizing total counts per cell...", flush=True)
    sc.pp.normalize_total(adata)

    print(f"[INFO][REF:{ref_name}] Applying log1p transformation...", flush=True)
    sc.pp.log1p(adata)

    print(f"[INFO][REF:{ref_name}] Finding top {NUM_HVGS} highly variable genes...", flush=True)
    sc.pp.highly_variable_genes(adata, n_top_genes=NUM_HVGS)

    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("[ERROR][REF] 'highly_variable' missing after HVG selection.")

    hvg_mask = adata.var["highly_variable"].values
    hvg_genes = adata.var_names[hvg_mask].astype(str).tolist()

    if len(hvg_genes) != NUM_HVGS:
        raise RuntimeError(
            f"[ERROR][REF] Expected {NUM_HVGS} HVGs, got {len(hvg_genes)}. "
            "Check that you have enough common genes and that HVG selection succeeded."
        )

    print(f"[INFO][REF:{ref_name}] Selected HVG genes (within common): {len(hvg_genes):,}", flush=True)

    # We also store X_hvg for reference output file consistency:
    adata.obsm["X_hvg"] = adata[:, hvg_mask].X.toarray()

    # Save reference processed file
    out_path = Path(OUTPUT_DIR) / f"{ref_name}{OUT_SUFFIX}"
    print(f"[INFO][REF:{ref_name}] Writing reference preprocessed AnnData to: {out_path}", flush=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"[OK][REF:{ref_name}] Saved.", flush=True)

    del adata
    return hvg_genes


# ----------------- per-dataset processing (other datasets) ----------------- #

def preprocess_other_dataset(path: str, ref_hvg_genes: list[str]):
    """
    Load other dataset, apply gene naming + harmonisation,
    subset to the reference HVG list (must be present), then normalize/log1p and store X_hvg.
    If the output _aligned.h5ad already exists, skip processing.
    """
    in_path = Path(path)
    name = in_path.stem
    ds_path_str = str(in_path)
    out_path = Path(OUTPUT_DIR) / f"{name}{OUT_SUFFIX}"

    if out_path.exists():
        print("\n" + "=" * 80, flush=True)
        print(f"[INFO][{name}] Output already exists: {out_path} — skipping.", flush=True)
        return

    print("\n" + "=" * 80, flush=True)
    print(f"[INFO][{name}] Loading dataset: {in_path}", flush=True)
    adata_raw = ad.read_h5ad(in_path)

    adata_raw = apply_gene_names(adata_raw, dataset_path=ds_path_str, name=name)
    adata = harmonise_one(adata_raw, name=name)
    del adata_raw

    # Ensure all ref HVGs are present
    missing = [g for g in ref_hvg_genes if g not in adata.var_names]
    if missing:
        print(f"[WARN][{name}] Missing {len(missing)} reference HVG genes.", flush=True)
        if len(missing) <= MAX_PRINT_MISSING:
            print(f"[WARN][{name}] Missing HVGs example: {missing}", flush=True)
        raise RuntimeError(
            f"[ERROR][{name}] Cannot align to reference HVG list because some genes are missing. "
            "Since you have 22k common genes, this likely indicates a naming mismatch."
        )

    print(f"[INFO][{name}] Subsetting to reference HVG genes: {len(ref_hvg_genes):,}", flush=True)
    adata = adata[:, ref_hvg_genes].copy()
    print(f"[INFO][{name}] After HVG subsetting: {adata.n_obs:,} × {adata.n_vars:,}", flush=True)

    # replicate snippet (on the HVG feature space)
    print(f"[INFO][{name}] Normalizing total counts per cell...", flush=True)
    sc.pp.normalize_total(adata)

    print(f"[INFO][{name}] Applying log1p transformation...", flush=True)
    sc.pp.log1p(adata)

    # Store HVG expression (which is now just X) as dense
    print(f"[INFO][{name}] Storing HVG expression in obsm['X_hvg'] (dense)...", flush=True)
    adata.obsm["X_hvg"] = adata.X.toarray()

    print(f"[INFO][{name}] Writing preprocessed AnnData to: {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"[OK][{name}] Done.", flush=True)

    del adata


# ----------------- main ----------------- #

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load common genes
    common_path = Path(COMMON_GENE_FILE)
    if not common_path.exists():
        raise FileNotFoundError(
            f"COMMON_GENE_FILE not found: {common_path}\n"
            "Run compute_common_genes.py first."
        )
    common_genes = [g.strip() for g in common_path.read_text().splitlines() if g.strip()]
    if len(common_genes) < NUM_HVGS:
        raise RuntimeError(
            f"[ERROR] Only {len(common_genes)} common genes, but NUM_HVGS={NUM_HVGS}. "
            "Need at least NUM_HVGS common genes to choose HVGs from within common space."
        )
    print(f"[INFO] Loaded {len(common_genes):,} common genes from {common_path}", flush=True)

    # 1) Process reference: common -> HVG selection -> save + return HVG list
    ref_hvg_genes = compute_ref_hvg_gene_list(common_genes)

    # 2) Process other datasets: subset to ref HVG gene list -> save
    for p in OTHER_PATHS:
        preprocess_other_dataset(p, ref_hvg_genes)

    print("\n[DONE] Reference processed and used to define HVG gene list; other datasets aligned to it.", flush=True)
    print(f"[NOTE] Shared feature space length: {len(ref_hvg_genes)} genes", flush=True)


if __name__ == "__main__":
    main()
