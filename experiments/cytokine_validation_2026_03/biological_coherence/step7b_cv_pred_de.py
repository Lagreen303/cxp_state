#!/usr/bin/env python3
"""
Step 7b: Compute DE on CV predictions (concatenated across folds)
per (cytokine, cell_type), analogous to step1_atlas_de.py.

Saves cv_pred_de_results.csv for downstream GSEA comparison.

Usage (submit as SLURM job on l4):
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step7b_cv_pred_de.py
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

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
CV_BASE = BASE / "cross_validation"
CV_FOLDS = [CV_BASE / f"run_2026_02_05_fold{i}/eval_last.ckpt/adata_pred.h5ad" for i in range(1, 10)]
OUTDIR = BASE / "biological_coherence"
OUTPUT_PATH = OUTDIR / "cv_pred_de_results.csv"
PERT_COL = "cytokine"
CELLTYPE_COL = "cell_type"
CONTROL = "PBS"


def _normalize_sparse_inplace(adata: ad.AnnData, target_sum: float = 1e4):
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


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def _is_counts(adata: ad.AnnData) -> bool:
    X = _to_dense(adata.X[:min(200, adata.n_obs)])
    return bool((X >= 0).all() and np.allclose(X, X.astype(int)))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"[cv pred DE] Results already exist at {OUTPUT_PATH}, skipping.")
        return

    print("[cv pred DE] Loading and concatenating CV fold predictions ...")
    cv_adatas = []
    for fp in CV_FOLDS:
        if fp.exists():
            print(f"  Loading {fp.parent.parent.name} ...")
            cv_adatas.append(ad.read_h5ad(fp))
    adata = ad.concat(cv_adatas, join="inner")
    del cv_adatas
    adata.obs_names_make_unique()
    print(f"  Combined shape: {adata.shape}")

    if _is_counts(adata):
        print("  Normalising (counts detected) ...")
        _normalize_sparse_inplace(adata)
        print("  Done.")

    # Clamp extreme values
    X = _to_dense(adata.X) if issparse(adata.X) else np.asarray(adata.X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -50, 50, out=X)
    adata.X = X

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
        print("[cv pred DE] No results computed!")
        return

    df = pd.concat(all_rows, ignore_index=True)
    df = df.rename(columns={"names": "gene", "logfoldchanges": "logFC_cv_pred",
                            "pvals": "pval_cv_pred", "pvals_adj": "fdr_cv_pred",
                            "scores": "score_cv_pred"})
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[cv pred DE] Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"  {df['cytokine'].nunique()} cytokines, {df['cell_type'].nunique()} cell types")


if __name__ == "__main__":
    main()
