#!/usr/bin/env python3
"""
Step 3: Predicted vs observed expression scatters for treated cells.
Uses non-imputed genes only.

Produces:
  - Per cell type: hexbin scatter of every (cell, gene) value, real vs predicted.
  - Aggregated per dataset: scatter coloured by cell type.

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step3_pred_vs_real.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
RUN = BASE / "run_2026_02_11_all"
OUTDIR = BASE / "biological_coherence"

EXTERNAL_DATASETS = {
    "GSE96583": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/GSE96583_batch2_IFNbeta_aligned_avg.h5ad"),
        "pred_h5ad": RUN / "predictions/GSE96583_batch2_IFNbeta_aligned_avg_predictions.h5ad",
        "perturbation": "IFN-beta",
        "pert_col": "cytokine",
        "control_label": "control",
    },
    "GSE178429": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg.h5ad"),
        "pred_h5ad": RUN / "predictions/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_predictions.h5ad",
        "perturbation": "IFN-gamma",
        "pert_col": "cytokine",
        "control_label": "control",
    },
}


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def make_percell_scatter(
    adata_real, adata_pred, cell_type, cytokine, dataset_name,
    pert_col, celltype_col, outdir,
):
    """Hexbin scatter of every (cell, gene) expression value for one cell type."""
    mask_real = (adata_real.obs[celltype_col] == cell_type) & (adata_real.obs[pert_col] == cytokine)
    mask_pred = (adata_pred.obs[celltype_col] == cell_type) & (adata_pred.obs[pert_col] == cytokine)

    n_cells = min(mask_real.sum(), mask_pred.sum())
    if n_cells < 3:
        return None

    X_real = _to_dense(adata_real[mask_real].X)[:n_cells]
    X_pred = _to_dense(adata_pred[mask_pred].X)[:n_cells]
    n_genes = X_real.shape[1]

    real_flat = X_real.ravel()
    pred_flat = X_pred.ravel()

    r_val = pearsonr(real_flat, pred_flat)[0]

    real_log = np.log1p(np.clip(real_flat, 0, None))
    pred_log = np.log1p(np.clip(pred_flat, 0, None))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real_log, pred_log, s=2, alpha=0.25, c="steelblue", edgecolors="none",
               rasterized=True)

    hi = max(real_log.max(), pred_log.max()) * 1.02
    ax.plot([0, hi], [0, hi], "--", c="red", lw=1, alpha=0.7)
    ax.set_xlim(-0.02 * hi, hi)
    ax.set_ylim(-0.02 * hi, hi)
    ax.set_xlabel("Real expression (log1p)")
    ax.set_ylabel("Predicted expression (log1p)")
    ax.set_title(
        f"{dataset_name}  {cell_type} — {cytokine}\n"
        f"{n_cells} cells \u00d7 {n_genes} non-imputed genes\n"
        f"Pearson r = {r_val:.3f}"
    )
    plt.tight_layout()

    ct_safe = cell_type.replace(" ", "_").replace("/", "-")
    fname = outdir / f"pred_vs_real_{dataset_name}_{ct_safe}_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def make_aggregated_scatter(
    adata_real, adata_pred, cytokine, dataset_name,
    pert_col, celltype_col, outdir,
    max_cells_per_ct: int = 500,
):
    """
    Aggregated scatter across all cell types for one dataset.
    Each point is one (cell, gene) value; colour = cell type.
    Subsamples cells per cell type to keep the plot readable.
    """
    cell_types = sorted(adata_real.obs[celltype_col].unique())
    n_ct = len(cell_types)
    cmap = cm.get_cmap("tab20", n_ct)

    all_real, all_pred, all_ct = [], [], []

    for i, ct in enumerate(cell_types):
        mask_r = (adata_real.obs[celltype_col] == ct) & (adata_real.obs[pert_col] == cytokine)
        mask_p = (adata_pred.obs[celltype_col] == ct) & (adata_pred.obs[pert_col] == cytokine)

        n = min(mask_r.sum(), mask_p.sum())
        if n < 3:
            continue

        X_r = _to_dense(adata_real[mask_r].X)[:n]
        X_p = _to_dense(adata_pred[mask_p].X)[:n]

        # Subsample cells for plotting
        if n > max_cells_per_ct:
            idx = np.random.choice(n, max_cells_per_ct, replace=False)
            X_r = X_r[idx]
            X_p = X_p[idx]

        real_flat = X_r.ravel()
        pred_flat = X_p.ravel()

        all_real.append(real_flat)
        all_pred.append(pred_flat)
        all_ct.append(np.full(len(real_flat), i, dtype=int))

    if not all_real:
        return None

    real_cat = np.concatenate(all_real)
    pred_cat = np.concatenate(all_pred)
    ct_cat = np.concatenate(all_ct)

    r_val = pearsonr(real_cat, pred_cat)[0]

    real_log = np.log1p(np.clip(real_cat, 0, None))
    pred_log = np.log1p(np.clip(pred_cat, 0, None))

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, ct in enumerate(cell_types):
        mask = ct_cat == i
        if mask.sum() == 0:
            continue
        ax.scatter(real_log[mask], pred_log[mask], s=2, alpha=0.2,
                   c=[cmap(i)], label=ct, edgecolors="none", rasterized=True)

    hi = max(real_log.max(), pred_log.max()) * 1.02
    ax.plot([0, hi], [0, hi], "--", c="black", lw=1, alpha=0.6)
    ax.set_xlim(-0.02 * hi, hi)
    ax.set_ylim(-0.02 * hi, hi)
    ax.set_xlabel("Real expression (log1p)")
    ax.set_ylabel("Predicted expression (log1p)")
    ax.set_title(
        f"{dataset_name} — {cytokine}  (all cell types, non-imputed genes)\n"
        f"Pearson r = {r_val:.3f}"
    )
    ax.legend(fontsize=7, markerscale=5, loc="upper left", framealpha=0.8)
    plt.tight_layout()

    fname = outdir / f"pred_vs_real_{dataset_name}_all_celltypes_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def main():
    expr_dir = OUTDIR / "plots" / "pred_vs_real_expression"
    expr_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        real_path = ds_info["real_h5ad"]
        pred_path = ds_info["pred_h5ad"]
        if not real_path.exists() or not pred_path.exists():
            print(f"[SKIP] {ds_name}: missing h5ad")
            continue

        print(f"Loading {ds_name} ...")
        adata_real = ad.read_h5ad(real_path)
        adata_pred = ad.read_h5ad(pred_path)

        if "is_imputed" in adata_real.var.columns:
            keep = ~adata_real.var["is_imputed"].astype(bool).values
            adata_real = adata_real[:, keep].copy()
            adata_pred = adata_pred[:, keep].copy()
            print(f"  Kept {keep.sum()} non-imputed genes")

        pert_col = ds_info["pert_col"]
        cyto = ds_info["perturbation"]
        celltype_col = "cell_type"

        if celltype_col not in adata_real.obs.columns:
            print(f"  [SKIP] {ds_name}: no '{celltype_col}' column")
            continue

        # Per cell-type scatters
        for ct in sorted(adata_real.obs[celltype_col].unique()):
            fname = make_percell_scatter(
                adata_real, adata_pred, ct, cyto, ds_name,
                pert_col, celltype_col, expr_dir,
            )
            if fname:
                print(f"  {ct}: scatter saved")

        # Aggregated scatter coloured by cell type
        fname_agg = make_aggregated_scatter(
            adata_real, adata_pred, cyto, ds_name,
            pert_col, celltype_col, expr_dir,
        )
        if fname_agg:
            print(f"  Aggregated scatter saved: {fname_agg.name}")

        del adata_real, adata_pred

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
