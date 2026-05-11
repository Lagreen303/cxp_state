#!/usr/bin/env python3
"""
Step 7: Mean-per-gene expression scatters (real vs predicted).
For each cell type, compute mean expression per gene across treated cells,
then scatter real mean vs predicted mean. Also produces aggregated version.

Uses non-imputed genes only.

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step7_mean_expression.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
RUN = BASE / "run_2026_02_11_all"
OUTDIR = BASE / "biological_coherence" / "plots" / "mean_expression"

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


def make_mean_scatter(real_mean, pred_mean, title, outpath):
    r_pearson = pearsonr(real_mean, pred_mean)[0]
    r_spearman = spearmanr(real_mean, pred_mean)[0]

    real_log = np.log1p(np.clip(real_mean, 0, None))
    pred_log = np.log1p(np.clip(pred_mean, 0, None))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real_log, pred_log, s=6, alpha=0.5, c="steelblue", edgecolors="none", rasterized=True)

    hi = max(real_log.max(), pred_log.max()) * 1.02
    ax.plot([0, hi], [0, hi], "--", c="red", lw=1, alpha=0.7)
    ax.set_xlim(-0.02 * hi, hi)
    ax.set_ylim(-0.02 * hi, hi)
    ax.set_xlabel("Real mean expression (log1p)")
    ax.set_ylabel("Predicted mean expression (log1p)")
    ax.set_title(f"{title}\nPearson r = {r_pearson:.3f}, Spearman r = {r_spearman:.3f}")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath.name}")
    return r_pearson, r_spearman


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        print(f"=== {ds_name} ===")
        adata_real = ad.read_h5ad(ds_info["real_h5ad"])
        adata_pred = ad.read_h5ad(ds_info["pred_h5ad"])

        if "is_imputed" in adata_real.var.columns:
            keep = ~adata_real.var["is_imputed"].astype(bool).values
            adata_real = adata_real[:, keep].copy()
            adata_pred = adata_pred[:, keep].copy()

        pert_col = ds_info["pert_col"]
        pert_label = ds_info["perturbation"]
        control_label = ds_info["control_label"]
        ct_col = "cell_type"

        if ct_col not in adata_pred.obs.columns:
            adata_pred.obs[ct_col] = adata_real.obs[ct_col].values
        if pert_col not in adata_pred.obs.columns:
            adata_pred.obs[pert_col] = adata_real.obs[pert_col].values

        mask_treated_real = adata_real.obs[pert_col] != control_label
        mask_treated_pred = adata_pred.obs[pert_col] != control_label

        cell_types = sorted(adata_real.obs.loc[mask_treated_real, ct_col].unique())
        all_real_means = []
        all_pred_means = []
        all_ct_labels = []

        for ct in cell_types:
            ct_mask_real = mask_treated_real & (adata_real.obs[ct_col] == ct)
            ct_mask_pred = mask_treated_pred & (adata_pred.obs[ct_col] == ct)

            n = min(ct_mask_real.sum(), ct_mask_pred.sum())
            if n < 3:
                print(f"  Skipping {ct} (n={n})")
                continue

            X_real = _to_dense(adata_real[ct_mask_real].X)[:n]
            X_pred = _to_dense(adata_pred[ct_mask_pred].X)[:n]

            real_mean = X_real.mean(axis=0)
            pred_mean = X_pred.mean(axis=0)

            r_p, r_s = make_mean_scatter(
                real_mean, pred_mean,
                f"{ds_name}  {ct} — {pert_label}\n{n} cells, {len(real_mean)} non-imputed genes",
                OUTDIR / f"mean_expr_{ds_name}_{ct.replace(' ', '_')}_{pert_label}.png",
            )
            summary_rows.append({
                "dataset": ds_name, "cell_type": ct, "cytokine": pert_label,
                "pearson_mean": r_p, "spearman_mean": r_s, "n_cells": n,
            })
            all_real_means.append(real_mean)
            all_pred_means.append(pred_mean)
            all_ct_labels.extend([ct] * len(real_mean))

        if all_real_means:
            agg_real = np.concatenate(all_real_means)
            agg_pred = np.concatenate(all_pred_means)
            make_mean_scatter(
                agg_real, agg_pred,
                f"{ds_name} — {pert_label} (all cell types pooled)\n{len(cell_types)} cell types, mean per gene",
                OUTDIR / f"mean_expr_{ds_name}_all_celltypes_{pert_label}.png",
            )

    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        csv_path = BASE / "biological_coherence" / "mean_expression_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary to {csv_path.name}")
        print(df.to_string(index=False))

    print("\nStep 7 complete.")


if __name__ == "__main__":
    main()
