#!/usr/bin/env python3
"""
Step 7c: Pseudobulk analysis — expression and delta (perturbation effect)
correlations for all 3 dataset pairs.

For each (cell_type, perturbation):
  1. Pseudobulk expression (treated): mean per gene, real vs pred
  2. Pseudobulk delta: (treated_mean - control_mean) per gene, real vs pred

Datasets:
  - Atlas real vs CV predictions (--mode atlas_cv)
  - GSE96583 real vs predictions  (--mode external, run locally)
  - GSE178429 real vs predictions (--mode external, run locally)

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step7c_pseudobulk.py --mode external
    # For atlas CV (submit as SLURM job):
    uv run python experiments/cytokine_validation_2026/biological_coherence/step7c_pseudobulk.py --mode atlas_cv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
RUN = BASE / "run_2026_02_11_all"
OUTDIR = BASE / "biological_coherence" / "plots" / "pseudobulk"

ATLAS_PATH = Path("/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad")
CV_BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/cross_validation")
CV_FOLDS = [CV_BASE / f"run_2026_02_05_fold{i}/eval_last.ckpt/adata_pred.h5ad" for i in range(1, 10)]

FOCUS_CYTOKINES = ["IFN-beta", "IFN-gamma", "TNF-alpha", "IL-6", "IL-4", "IL-10"]

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


def _safe_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return np.nan, np.nan
    return pearsonr(a[mask], b[mask])[0], spearmanr(a[mask], b[mask])[0]


def make_scatter(x, y, xlabel, ylabel, title, outpath, log_transform=True, n_cells_suffix: str = ""):
    if log_transform:
        x = np.log1p(np.clip(x, 0, None))
        y = np.log1p(np.clip(y, 0, None))

    r_p, r_s = _safe_corr(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=6, alpha=0.5, c="steelblue", edgecolors="none", rasterized=True)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    margin = (hi - lo) * 0.02
    ax.plot([lo, hi], [lo, hi], "--", c="red", lw=1, alpha=0.7)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title_line = f"{title}\nPearson r = {r_p:.3f}, Spearman r = {r_s:.3f}"
    if n_cells_suffix:
        title_line += f"\n{n_cells_suffix}"
    ax.set_title(title_line)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {outpath.name}")
    return r_p, r_s


def pseudobulk_external(ds_name, ds_info, outdir):
    print(f"\n=== Pseudobulk: {ds_name} ===")
    adata_real = ad.read_h5ad(ds_info["real_h5ad"])
    adata_pred = ad.read_h5ad(ds_info["pred_h5ad"])

    if "is_imputed" in adata_real.var.columns:
        keep = ~adata_real.var["is_imputed"].astype(bool).values
        adata_real = adata_real[:, keep].copy()
        adata_pred = adata_pred[:, keep].copy()

    pert_col = ds_info["pert_col"]
    pert_label = ds_info["perturbation"]
    ctrl_label = ds_info["control_label"]
    ct_col = "cell_type"

    if ct_col not in adata_pred.obs.columns:
        adata_pred.obs[ct_col] = adata_real.obs[ct_col].values
    if pert_col not in adata_pred.obs.columns:
        adata_pred.obs[pert_col] = adata_real.obs[pert_col].values

    rows = []
    cell_types = sorted(adata_real.obs[ct_col].unique())

    for ct in cell_types:
        ctrl_real = (adata_real.obs[ct_col] == ct) & (adata_real.obs[pert_col] == ctrl_label)
        trt_real = (adata_real.obs[ct_col] == ct) & (adata_real.obs[pert_col] != ctrl_label)
        ctrl_pred = (adata_pred.obs[ct_col] == ct) & (adata_pred.obs[pert_col] == ctrl_label)
        trt_pred = (adata_pred.obs[ct_col] == ct) & (adata_pred.obs[pert_col] != ctrl_label)

        n_ctrl = min(ctrl_real.sum(), ctrl_pred.sum())
        n_trt = min(trt_real.sum(), trt_pred.sum())
        if n_ctrl < 3 or n_trt < 3:
            print(f"  Skipping {ct} (ctrl={n_ctrl}, trt={n_trt})")
            continue

        real_ctrl_mean = _to_dense(adata_real[ctrl_real].X).mean(axis=0)
        real_trt_mean = _to_dense(adata_real[trt_real].X).mean(axis=0)
        pred_ctrl_mean = _to_dense(adata_pred[ctrl_pred].X).mean(axis=0)
        pred_trt_mean = _to_dense(adata_pred[trt_pred].X).mean(axis=0)

        real_delta = real_trt_mean - real_ctrl_mean
        pred_delta = pred_trt_mean - pred_ctrl_mean

        safe_ct = ct.replace(" ", "_")

        n_suffix = f"n_ctrl={n_ctrl:,}, n_trt={n_trt:,}"
        r_expr_p, r_expr_s = make_scatter(
            real_trt_mean, pred_trt_mean,
            "Real mean expr (log1p)", "Predicted mean expr (log1p)",
            f"{ds_name}  {ct} — {pert_label}\nPseudobulk expression (treated)",
            outdir / f"pb_expr_{ds_name}_{safe_ct}_{pert_label}.png",
            n_cells_suffix=n_suffix,
        )

        r_delta_p, r_delta_s = make_scatter(
            real_delta, pred_delta,
            "Real delta (treated - control)", "Predicted delta",
            f"{ds_name}  {ct} — {pert_label}\nPseudobulk delta (perturbation effect)",
            outdir / f"pb_delta_{ds_name}_{safe_ct}_{pert_label}.png",
            log_transform=False,
            n_cells_suffix=n_suffix,
        )

        rows.append({
            "dataset": ds_name, "cell_type": ct, "cytokine": pert_label,
            "r_expr_pearson": r_expr_p, "r_expr_spearman": r_expr_s,
            "r_delta_pearson": r_delta_p, "r_delta_spearman": r_delta_s,
            "n_ctrl": int(n_ctrl), "n_trt": int(n_trt),
        })

    return rows


def pseudobulk_atlas_cv(outdir):
    print("\n=== Pseudobulk: Atlas real vs CV predictions ===")
    print("Loading atlas ...")
    atlas = ad.read_h5ad(ATLAS_PATH)
    print(f"  Atlas shape: {atlas.shape}")

    print("Loading CV predictions ...")
    cv_adatas = []
    for fp in CV_FOLDS:
        if fp.exists():
            cv_adatas.append(ad.read_h5ad(fp))
    cv_pred = ad.concat(cv_adatas, join="inner")
    del cv_adatas
    print(f"  CV pred shape: {cv_pred.shape}")

    shared_genes = atlas.var_names.intersection(cv_pred.var_names)
    atlas = atlas[:, shared_genes].copy()
    cv_pred = cv_pred[:, shared_genes].copy()

    atlas_cytokines = atlas.obs["cytokine"].unique()
    focus = [c for c in FOCUS_CYTOKINES if c in atlas_cytokines]
    if not focus:
        focus = list(atlas_cytokines[:6])
    print(f"  Focus cytokines: {focus}")

    ct_col = "cell_type"
    pert_col = "cytokine"
    ctrl_label = "PBS"
    rows = []

    for cyto in focus:
        print(f"\n  Cytokine: {cyto}")
        cell_types = sorted(set(atlas.obs[ct_col].unique()) & set(cv_pred.obs[ct_col].unique()))

        for ct in cell_types:
            ctrl_real = (atlas.obs[ct_col] == ct) & (atlas.obs[pert_col] == ctrl_label)
            trt_real = (atlas.obs[ct_col] == ct) & (atlas.obs[pert_col] == cyto)
            ctrl_pred = (cv_pred.obs[ct_col] == ct) & (cv_pred.obs[pert_col] == ctrl_label)
            trt_pred = (cv_pred.obs[ct_col] == ct) & (cv_pred.obs[pert_col] == cyto)

            n_ctrl_r, n_trt_r = int(ctrl_real.sum()), int(trt_real.sum())
            n_ctrl_p, n_trt_p = int(ctrl_pred.sum()), int(trt_pred.sum())
            n_ctrl, n_trt = min(n_ctrl_r, n_ctrl_p), min(n_trt_r, n_trt_p)
            if min(n_ctrl_r, n_trt_r, n_ctrl_p, n_trt_p) < 3:
                continue

            n_suffix = f"n_ctrl={n_ctrl:,}, n_trt={n_trt:,}"
            real_ctrl_mean = _to_dense(atlas[ctrl_real].X).mean(axis=0)
            real_trt_mean = _to_dense(atlas[trt_real].X).mean(axis=0)
            pred_ctrl_mean = _to_dense(cv_pred[ctrl_pred].X).mean(axis=0)
            pred_trt_mean = _to_dense(cv_pred[trt_pred].X).mean(axis=0)

            real_delta = real_trt_mean - real_ctrl_mean
            pred_delta = pred_trt_mean - pred_ctrl_mean

            safe_ct = ct.replace(" ", "_")
            safe_cyto = cyto.replace(" ", "_")

            r_expr_p, r_expr_s = make_scatter(
                real_trt_mean, pred_trt_mean,
                "Real mean expr (log1p)", "CV pred mean expr (log1p)",
                f"Atlas CV  {ct} — {cyto}\nPseudobulk expression (treated)",
                outdir / f"pb_expr_atlas_cv_{safe_ct}_{safe_cyto}.png",
                n_cells_suffix=n_suffix,
            )

            r_delta_p, r_delta_s = make_scatter(
                real_delta, pred_delta,
                "Real delta (treated - PBS)", "CV pred delta",
                f"Atlas CV  {ct} — {cyto}\nPseudobulk delta (perturbation effect)",
                outdir / f"pb_delta_atlas_cv_{safe_ct}_{safe_cyto}.png",
                log_transform=False,
                n_cells_suffix=n_suffix,
            )

            rows.append({
                "dataset": "atlas_cv", "cell_type": ct, "cytokine": cyto,
                "r_expr_pearson": r_expr_p, "r_expr_spearman": r_expr_s,
                "r_delta_pearson": r_delta_p, "r_delta_spearman": r_delta_s,
                "n_ctrl": min(n_ctrl_r, n_ctrl_p), "n_trt": min(n_trt_r, n_trt_p),
            })
            print(f"    {ct}: expr r={r_expr_p:.3f}, delta r={r_delta_p:.3f}")

    return rows


def make_summary_barplot(df, metric_col, ylabel, title, outpath):
    datasets = df["dataset"].unique()
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.5), 5))

    x_pos = 0
    xticks, xlabels = [], []
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    for i, ds in enumerate(sorted(datasets)):
        sub = df[df["dataset"] == ds].sort_values("cell_type")
        for _, row in sub.iterrows():
            val = row[metric_col]
            if np.isnan(val):
                continue
            ax.bar(x_pos, val, color=colors[i], edgecolor="black", linewidth=0.5)
            xticks.append(x_pos)
            lbl = f"{row['cell_type']}\n({ds})"
            if "n_ctrl" in row and "n_trt" in row and pd.notna(row["n_ctrl"]) and pd.notna(row["n_trt"]):
                n_tot = int(row["n_ctrl"]) + int(row["n_trt"])
                lbl += f"\nn={n_tot:,}"
            xlabels.append(lbl)
            x_pos += 1
        x_pos += 0.5

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.5)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["external", "atlas_cv", "all"], default="all")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    if args.mode in ("external", "all"):
        for ds_name, ds_info in EXTERNAL_DATASETS.items():
            rows = pseudobulk_external(ds_name, ds_info, OUTDIR)
            all_rows.extend(rows)

    if args.mode in ("atlas_cv", "all"):
        rows = pseudobulk_atlas_cv(OUTDIR)
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_path = BASE / "biological_coherence" / "pseudobulk_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} rows to {csv_path.name}")
        print(df.to_string(index=False))

        make_summary_barplot(
            df, "r_delta_pearson", "Pearson r (delta)",
            "Pseudobulk delta correlation (perturbation effect)",
            OUTDIR / "summary_delta_r_barplot.png",
        )
        make_summary_barplot(
            df, "r_expr_pearson", "Pearson r (expression)",
            "Pseudobulk expression correlation (treated cells)",
            OUTDIR / "summary_expr_r_barplot.png",
        )

    print("\nStep 7c complete.")


if __name__ == "__main__":
    main()
