#!/usr/bin/env python3
"""
Step 1: Compute DE per (cytokine, cell_type) for the cytokine atlas.

Processes one cell type at a time to limit memory.
Saves results to atlas_de_results.csv for downstream coherence analyses.

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step1_atlas_de.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=FutureWarning)

ATLAS_PATH = Path("/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad")
OUTDIR = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/biological_coherence")
OUTPUT_PATH = OUTDIR / "atlas_de_results.csv"
PERT_COL = "cytokine"
CELLTYPE_COL = "cell_type"
CONTROL = "PBS"


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def _normalize_sparse_inplace(adata: ad.AnnData, target_sum: float = 1e4):
    """Vectorized per-cell normalization + log1p."""
    from scipy.sparse import diags
    X = adata.X
    if issparse(X):
        X = X.tocsr()
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        scaling = target_sum / row_sums
        adata.X = diags(scaling) @ X
    else:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adata.X = X / row_sums * target_sum
    sc.pp.log1p(adata)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"[atlas DE] Results already exist at {OUTPUT_PATH}, skipping.")
        return

    print(f"[atlas DE] Loading atlas from {ATLAS_PATH} ...")
    adata = ad.read_h5ad(ATLAS_PATH)
    print(f"  shape: {adata.shape}")

    # Detect counts and normalize
    X_sample = _to_dense(adata.X[:100])
    is_counts = (X_sample >= 0).all() and np.allclose(X_sample, X_sample.astype(int))
    if is_counts:
        print("  Normalising atlas (counts detected) ...")
        _normalize_sparse_inplace(adata)
        print("  Normalisation done.")

    cell_types = sorted(adata.obs[CELLTYPE_COL].unique())
    cytokines = sorted([c for c in adata.obs[PERT_COL].unique() if c != CONTROL])
    print(f"  {len(cell_types)} cell types, {len(cytokines)} cytokines (excl. {CONTROL})")

    all_rows = []
    for ct in cell_types:
        ct_mask = adata.obs[CELLTYPE_COL] == ct
        ct_ctrl_mask = ct_mask & (adata.obs[PERT_COL] == CONTROL)
        n_ctrl = ct_ctrl_mask.sum()
        if n_ctrl < 3:
            print(f"  {ct}: <3 control cells, skipping")
            continue

        ct_count = 0
        for cyto in cytokines:
            pert_mask = ct_mask & (adata.obs[PERT_COL] == cyto)
            n_pert = pert_mask.sum()
            if n_pert < 3:
                continue

            ctrl_idx = np.where(ct_ctrl_mask.values)[0]
            pert_idx = np.where(pert_mask.values)[0]
            idx = np.concatenate([ctrl_idx, pert_idx])
            sub = adata[idx].copy()
            sub.obs["_group"] = "control"
            sub.obs.iloc[len(ctrl_idx):, sub.obs.columns.get_loc("_group")] = "perturbed"

            try:
                sc.tl.rank_genes_groups(sub, groupby="_group", groups=["perturbed"],
                                        reference="control", method="wilcoxon",
                                        use_raw=False)
            except Exception as e:
                print(f"  [WARN] DE failed for {ct}/{cyto}: {e}")
                del sub
                continue

            result = sc.get.rank_genes_groups_df(sub, group="perturbed")
            result["cell_type"] = ct
            result["cytokine"] = cyto
            all_rows.append(result)
            del sub
            ct_count += 1

        print(f"  Done: {ct} ({ct_count} cytokines)")

    if not all_rows:
        print("[atlas DE] No results computed!")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df = df.rename(columns={"names": "gene", "logfoldchanges": "logFC_atlas",
                            "pvals": "pval_atlas", "pvals_adj": "fdr_atlas",
                            "scores": "score_atlas"})
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[atlas DE] Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"  {df['cytokine'].nunique()} cytokines, {df['cell_type'].nunique()} cell types")


if __name__ == "__main__":
    main()
