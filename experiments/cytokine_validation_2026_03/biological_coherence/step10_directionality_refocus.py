#!/usr/bin/env python3
"""
Phase 1 refocus: recompute DE tables and directionality-first metrics.

Outputs are written into a dated folder, separate from historical outputs:
  biological_coherence/de_directionality_YYYY-MM-DD/
    - de_tables/
    - directionality_metrics/
    - manifest.yaml
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy.sparse import issparse


BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026_03")
BIO = BASE / "biological_coherence"
RUN = BASE / "run_2026_02_11_all"
DATE_TAG = dt.date.today().isoformat()
OUT = BIO / f"de_directionality_{DATE_TAG}"
DE_DIR = OUT / "de_tables"
MET_DIR = OUT / "directionality_metrics"

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


def _normalize_sparse_inplace(adata: ad.AnnData, target_sum: float = 1e4):
    from scipy.sparse import diags

    X = adata.X
    if issparse(X):
        X = X.tocsr()
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        adata.X = diags(target_sum / row_sums) @ X
    else:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adata.X = X / row_sums * target_sum
    sc.pp.log1p(adata)


def _is_counts(adata: ad.AnnData) -> bool:
    X = _to_dense(adata.X[: min(200, adata.n_obs)])
    return bool((X >= 0).all() and np.allclose(X, X.astype(int)))


def compute_de_for_celltype(
    adata: ad.AnnData,
    *,
    cell_type: str,
    cytokine: str,
    pert_col: str,
    control_label: str,
) -> pd.DataFrame | None:
    ct_col = "cell_type"
    mask_ctrl = (adata.obs[ct_col] == cell_type) & (adata.obs[pert_col] == control_label)
    mask_trt = (adata.obs[ct_col] == cell_type) & (adata.obs[pert_col] == cytokine)
    n_ctrl = int(mask_ctrl.sum())
    n_trt = int(mask_trt.sum())
    if n_ctrl < 3 or n_trt < 3:
        return None

    idx = np.concatenate([np.where(mask_ctrl.values)[0], np.where(mask_trt.values)[0]])
    sub = adata[idx].copy()
    sub.obs["_group"] = "control"
    sub.obs.iloc[n_ctrl:, sub.obs.columns.get_loc("_group")] = "perturbed"

    sc.tl.rank_genes_groups(
        sub,
        groupby="_group",
        groups=["perturbed"],
        reference="control",
        method="wilcoxon",
        use_raw=False,
    )
    df = sc.get.rank_genes_groups_df(sub, group="perturbed")
    df["cell_type"] = cell_type
    df["cytokine"] = cytokine
    df["n_ctrl"] = n_ctrl
    df["n_trt"] = n_trt
    return df


def direction_metrics(merged: pd.DataFrame) -> dict[str, float]:
    # all genes first (requested): compare signs on all shared genes
    s_real = np.sign(merged["logFC_real"].values)
    s_pred = np.sign(merged["logFC_pred"].values)
    all_sign_match = float((s_real == s_pred).mean())

    # non-zero-only sign agreement (auxiliary)
    nz = (s_real != 0) & (s_pred != 0)
    nz_sign_match = float(((s_real[nz] == s_pred[nz]).mean()) if nz.sum() > 0 else np.nan)

    # DE subset after all-genes metrics (requested second)
    de_real = merged["fdr_real"].values < 0.05
    de_pred = merged["fdr_pred"].values < 0.05
    de_union = de_real | de_pred
    de_inter = de_real & de_pred

    def _sign_rate(mask):
        if mask.sum() == 0:
            return np.nan
        return float((s_real[mask] == s_pred[mask]).mean())

    return {
        "n_shared_genes": int(len(merged)),
        "all_genes_sign_match": all_sign_match,
        "nonzero_genes_sign_match": nz_sign_match,
        "n_de_real_fdr05": int(de_real.sum()),
        "n_de_pred_fdr05": int(de_pred.sum()),
        "n_de_union_fdr05": int(de_union.sum()),
        "n_de_intersection_fdr05": int(de_inter.sum()),
        "de_real_sign_match": _sign_rate(de_real),
        "de_union_sign_match": _sign_rate(de_union),
        "de_intersection_sign_match": _sign_rate(de_inter),
    }


def main():
    DE_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)

    all_real_rows = []
    all_pred_rows = []
    metric_rows = []
    skipped_rows = []

    for ds_name, ds in EXTERNAL_DATASETS.items():
        print(f"[{ds_name}] loading real/pred h5ad ...")
        adata_real = ad.read_h5ad(ds["real_h5ad"])
        adata_pred = ad.read_h5ad(ds["pred_h5ad"])

        if "is_imputed" in adata_real.var.columns:
            keep = ~adata_real.var["is_imputed"].astype(bool).values
            adata_real = adata_real[:, keep].copy()
            adata_pred = adata_pred[:, keep].copy()

        # Ensure required obs columns exist on prediction object
        for col in ["cell_type", ds["pert_col"]]:
            if col not in adata_pred.obs.columns and col in adata_real.obs.columns:
                adata_pred.obs[col] = adata_real.obs[col].values

        if _is_counts(adata_real):
            print(f"[{ds_name}] real appears count-like; normalizing/log1p for DE.")
            _normalize_sparse_inplace(adata_real)

        cyto = ds["perturbation"]
        pert_col = ds["pert_col"]
        ctrl = ds["control_label"]
        cts = sorted(set(adata_real.obs["cell_type"].unique()) & set(adata_pred.obs["cell_type"].unique()))

        for ct in cts:
            de_real = compute_de_for_celltype(
                adata_real,
                cell_type=ct,
                cytokine=cyto,
                pert_col=pert_col,
                control_label=ctrl,
            )
            de_pred = compute_de_for_celltype(
                adata_pred,
                cell_type=ct,
                cytokine=cyto,
                pert_col=pert_col,
                control_label=ctrl,
            )
            if de_real is None or de_pred is None:
                skipped_rows.append(
                    {"dataset": ds_name, "cell_type": ct, "cytokine": cyto, "reason": "insufficient_cells_or_de_failure"}
                )
                continue

            real = de_real.rename(
                columns={
                    "names": "gene",
                    "scores": "score_real",
                    "logfoldchanges": "logFC_real",
                    "pvals": "pval_real",
                    "pvals_adj": "fdr_real",
                }
            )
            pred = de_pred.rename(
                columns={
                    "names": "gene",
                    "scores": "score_pred",
                    "logfoldchanges": "logFC_pred",
                    "pvals": "pval_pred",
                    "pvals_adj": "fdr_pred",
                }
            )
            real["dataset"] = ds_name
            pred["dataset"] = ds_name
            all_real_rows.append(real)
            all_pred_rows.append(pred)

            merged = (
                real[["gene", "logFC_real", "fdr_real"]]
                .merge(pred[["gene", "logFC_pred", "fdr_pred"]], on="gene", how="inner")
                .dropna()
            )
            if len(merged) == 0:
                skipped_rows.append({"dataset": ds_name, "cell_type": ct, "cytokine": cyto, "reason": "no_shared_genes_after_merge"})
                continue

            row = {"dataset": ds_name, "cell_type": ct, "cytokine": cyto}
            row.update(direction_metrics(merged))
            metric_rows.append(row)

            # condition-specific DE outputs
            ct_safe = ct.replace(" ", "_").replace("/", "-")
            base = f"{ds_name}_{ct_safe}_{cyto.replace(' ', '_')}"
            real.to_csv(DE_DIR / f"de_real_{base}.csv", index=False)
            pred.to_csv(DE_DIR / f"de_pred_{base}.csv", index=False)
            merged.to_csv(DE_DIR / f"de_merged_{base}.csv", index=False)

    if all_real_rows:
        pd.concat(all_real_rows, ignore_index=True).to_csv(DE_DIR / "de_real_all.csv", index=False)
    if all_pred_rows:
        pd.concat(all_pred_rows, ignore_index=True).to_csv(DE_DIR / "de_pred_all.csv", index=False)
    if metric_rows:
        mdf = pd.DataFrame(metric_rows).sort_values(["dataset", "cell_type"])
        mdf.to_csv(MET_DIR / "directionality_metrics_all_conditions.csv", index=False)

        # easy scan plot (all-genes first + DE subset second)
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(mdf) * 0.35)))
        labels = [f"{r.cell_type}\n({r.dataset})" for r in mdf.itertuples()]
        y = np.arange(len(mdf))
        axes[0].barh(y, mdf["all_genes_sign_match"].values * 100, color="#2b8cbe")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].set_xlim(0, 100)
        axes[0].set_xlabel("Sign match %")
        axes[0].set_title("All genes (primary)")

        axes[1].barh(y, np.nan_to_num(mdf["de_real_sign_match"].values) * 100, color="#de2d26")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Sign match %")
        axes[1].set_title("DE-only (real FDR<0.05)")
        plt.tight_layout()
        fig.savefig(MET_DIR / "directionality_sign_match_primary_vs_de.png", dpi=220)
        plt.close(fig)

    pd.DataFrame(skipped_rows).to_csv(MET_DIR / "skipped_conditions.csv", index=False)

    manifest = {
        "date": DATE_TAG,
        "phase": "phase1_directionality_refocus",
        "model_run": str(RUN),
        "datasets": {k: {kk: str(vv) for kk, vv in d.items()} for k, d in EXTERNAL_DATASETS.items()},
        "outputs": {
            "root": str(OUT),
            "de_tables": str(DE_DIR),
            "directionality_metrics": str(MET_DIR),
        },
        "notes": {
            "strategy": "all_genes_directionality_first_then_de_subset",
            "de_method": "scanpy.rank_genes_groups_wilcoxon",
            "de_subset_definition": "real_fdr<0.05 plus union/intersection side metrics",
        },
    }
    with open(OUT / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"[DONE] Phase 1 outputs written to: {OUT}")


if __name__ == "__main__":
    main()

