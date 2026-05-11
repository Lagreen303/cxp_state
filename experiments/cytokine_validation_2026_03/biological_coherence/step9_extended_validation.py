#!/usr/bin/env python3
"""
Step 9: Extended validation analyses.

Six additional analyses built on existing metric CSVs and h5ad files:
  1. Effect-size calibration   (OLS slope on real vs predicted logFC)
  2. Direction accuracy         (% genes with matching logFC sign)
  3. Top-k pathway recovery     (precision@k from GSEA results)
  4. Distribution-level metric  (per-gene Wasserstein distance)
  5. Failure-mode stratification (performance vs cell count / effect size)
  6. Master summary heatmap     (all metrics in one figure)

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step9_extended_validation.py
    # Or individual analyses:
    uv run python ... --analysis calibration
    uv run python ... --analysis direction
    uv run python ... --analysis topk_pathway
    uv run python ... --analysis distribution
    uv run python ... --analysis failure_modes
    uv run python ... --analysis master_heatmap
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ks_2samp

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
BIO_DIR = BASE / "biological_coherence"
OUTDIR = BIO_DIR / "plots" / "extended"
RUN = BASE / "run_2026_02_11_all"

ATLAS_DE_PATH = BIO_DIR / "atlas_de_results.csv"
COHERENCE_PATH = BIO_DIR / "gene_coherence_metrics.csv"
MEAN_EXPR_PATH = BIO_DIR / "mean_expression_metrics.csv"
PSEUDOBULK_PATH = BIO_DIR / "pseudobulk_metrics.csv"
GSEA_DIR = BIO_DIR / "plots" / "gsea"

EXTERNAL_DATASETS = {
    "GSE96583": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/GSE96583_batch2_IFNbeta_aligned_avg.h5ad"),
        "pred_h5ad": RUN / "predictions/GSE96583_batch2_IFNbeta_aligned_avg_predictions.h5ad",
        "atlas_cytokine": "IFN-beta",
        "pert_col": "cytokine",
        "control_label": "control",
    },
    "GSE178429": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg.h5ad"),
        "pred_h5ad": RUN / "predictions/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_predictions.h5ad",
        "atlas_cytokine": "IFN-gamma",
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


def compute_de_inline(adata, pert_col, control_label, cell_type):
    """Compute DE for a single cell type (returns full DataFrame with logFC + FDR)."""
    ct_col = "cell_type"
    mask = adata.obs[ct_col] == cell_type
    sub = adata[mask].copy()

    ctrl_mask = sub.obs[pert_col] == control_label
    trt_mask = ~ctrl_mask
    if ctrl_mask.sum() < 3 or trt_mask.sum() < 3:
        return None

    sub.obs["_group"] = "control"
    sub.obs.loc[trt_mask, "_group"] = "perturbed"

    try:
        sc.tl.rank_genes_groups(sub, groupby="_group", groups=["perturbed"],
                                reference="control", method="wilcoxon", use_raw=False)
    except Exception:
        return None

    return sc.get.rank_genes_groups_df(sub, group="perturbed")


def _load_external_pair(ds_name: str):
    """Load and align real / predicted h5ad for an external dataset."""
    ds_info = EXTERNAL_DATASETS[ds_name]
    adata_real = ad.read_h5ad(ds_info["real_h5ad"])
    adata_pred = ad.read_h5ad(ds_info["pred_h5ad"])

    if "is_imputed" in adata_real.var.columns:
        keep = ~adata_real.var["is_imputed"].astype(bool).values
        adata_real = adata_real[:, keep].copy()
        adata_pred = adata_pred[:, keep].copy()

    ct_col, pert_col = "cell_type", ds_info["pert_col"]
    if ct_col not in adata_pred.obs.columns:
        adata_pred.obs[ct_col] = adata_real.obs[ct_col].values
    if pert_col not in adata_pred.obs.columns:
        adata_pred.obs[pert_col] = adata_real.obs[pert_col].values

    return adata_real, adata_pred, ds_info


def _get_logfc_pairs():
    """Yield (dataset, cell_type, cytokine, logfc_real, logfc_pred, fdr_real) tuples.

    Compares external-real DE vs external-predicted DE (Level-3 style).
    """
    for ds_name in EXTERNAL_DATASETS:
        adata_real, adata_pred, ds_info = _load_external_pair(ds_name)
        cyto = ds_info["atlas_cytokine"]
        pert_col = ds_info["pert_col"]
        ctrl = ds_info["control_label"]

        X_sample = _to_dense(adata_real.X[:100])
        if (X_sample >= 0).all() and np.allclose(X_sample, X_sample.astype(int)):
            _normalize_sparse_inplace(adata_real)

        cell_types = sorted(adata_real.obs["cell_type"].unique())
        for ct in cell_types:
            de_real = compute_de_inline(adata_real, pert_col, ctrl, ct)
            de_pred = compute_de_inline(adata_pred, pert_col, ctrl, ct)
            if de_real is None or de_pred is None:
                continue

            real = de_real.set_index("names")[["logfoldchanges", "pvals_adj"]].rename(
                columns={"logfoldchanges": "logFC_real", "pvals_adj": "fdr_real"}
            )
            pred = de_pred.set_index("names")[["logfoldchanges"]].rename(
                columns={"logfoldchanges": "logFC_pred"}
            )
            merged = real.join(pred, how="inner").dropna()
            if len(merged) < 20:
                continue

            yield ds_name, ct, cyto, merged["logFC_real"].values, merged["logFC_pred"].values, merged["fdr_real"].values


# ---------------------------------------------------------------------------
# 1. Effect-size calibration
# ---------------------------------------------------------------------------

def analysis_calibration(outdir: Path) -> pd.DataFrame:
    print("\n=== Effect-size calibration ===")
    rows = []

    for ds, ct, cyto, lfc_real, lfc_pred, _fdr in _get_logfc_pairs():
        from numpy.polynomial.polynomial import polyfit
        coeffs = polyfit(lfc_real, lfc_pred, 1)
        intercept, slope = coeffs[0], coeffs[1]
        ss_res = np.sum((lfc_pred - (intercept + slope * lfc_real)) ** 2)
        ss_tot = np.sum((lfc_pred - lfc_pred.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append({
            "dataset": ds, "cell_type": ct, "cytokine": cyto,
            "slope": slope, "intercept": intercept, "r_squared": r_squared,
            "n_genes": len(lfc_real),
        })

        safe = f"{ds}_{ct}_{cyto}".replace(" ", "_")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(lfc_real, lfc_pred, s=6, alpha=0.4, c="steelblue",
                   edgecolors="none", rasterized=True)
        lo, hi = min(lfc_real.min(), lfc_pred.min()), max(lfc_real.max(), lfc_pred.max())
        margin = (hi - lo) * 0.05
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, intercept + slope * xs, c="darkorange", lw=1.5, label=f"OLS: y = {slope:.2f}x + {intercept:.2f}")
        ax.plot([lo, hi], [lo, hi], "--", c="grey", lw=1, alpha=0.5)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel("Real logFC")
        ax.set_ylabel("Predicted logFC")
        ax.set_title(f"Effect-size calibration: {ct} — {cyto}\n"
                     f"slope = {slope:.3f}, R² = {r_squared:.3f}, n = {len(lfc_real)}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        fig.savefig(outdir / f"calibration_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  {ds} {ct}: slope={slope:.3f}, R²={r_squared:.3f}")

    if not rows:
        print("  No conditions computed.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.6), 5))
    labels = [f"{r['cell_type']}\n({r['dataset']})" for _, r in df.iterrows()]
    colors = np.where(np.abs(df["slope"].values - 1) < 0.3, "#4dac26", "#d6604d")
    bars = ax.bar(range(len(df)), df["slope"].values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="red", ls="--", lw=1, label="Ideal slope = 1")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("OLS slope (predicted vs real logFC)")
    ax.set_title("Effect-size calibration across conditions")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "calibration_summary_slopes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved calibration_summary_slopes.png")

    return df


# ---------------------------------------------------------------------------
# 2. Direction accuracy
# ---------------------------------------------------------------------------

def analysis_direction(outdir: Path) -> pd.DataFrame:
    print("\n=== Direction accuracy ===")
    rows = []

    for ds, ct, cyto, lfc_real, lfc_pred, fdr_real in _get_logfc_pairs():
        nonzero = (lfc_real != 0) & (lfc_pred != 0)
        if nonzero.sum() < 10:
            continue

        same_sign_all = np.sign(lfc_real[nonzero]) == np.sign(lfc_pred[nonzero])
        dir_acc_all = same_sign_all.mean()

        de_mask = nonzero & (fdr_real < 0.05)
        if de_mask.sum() >= 5:
            same_sign_de = np.sign(lfc_real[de_mask]) == np.sign(lfc_pred[de_mask])
            dir_acc_de = same_sign_de.mean()
            n_de = int(de_mask.sum())
        else:
            dir_acc_de = np.nan
            n_de = int(de_mask.sum())

        rows.append({
            "dataset": ds, "cell_type": ct, "cytokine": cyto,
            "dir_acc_all": dir_acc_all, "dir_acc_de": dir_acc_de,
            "n_all": int(nonzero.sum()), "n_de": n_de,
        })
        print(f"  {ds} {ct}: all={dir_acc_all:.1%}, DE={dir_acc_de:.1%} (n_de={n_de})")

    if not rows:
        print("  No conditions computed.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.7), 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["dir_acc_all"].values * 100, w, label="All genes",
           color="#2166ac", edgecolor="black", linewidth=0.5)
    de_vals = df["dir_acc_de"].fillna(0).values * 100
    ax.bar(x + w / 2, de_vals, w, label="DE genes (FDR < 0.05)",
           color="#b2182b", edgecolor="black", linewidth=0.5)
    ax.axhline(50, color="grey", ls="--", lw=1, alpha=0.7, label="Random (50%)")
    labels = [f"{r['cell_type']}\n({r['dataset']})" for _, r in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Direction accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Gene-level direction accuracy (real vs predicted logFC)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "direction_accuracy_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved direction_accuracy_bar.png")

    return df


# ---------------------------------------------------------------------------
# 3. Top-k pathway recovery
# ---------------------------------------------------------------------------

def _parse_source(source: str):
    for prefix, side in [
        ("ext_real_", "real"), ("ext_pred_", "pred"),
        ("atlas_", "real"), ("cv_pred_", "pred"),
    ]:
        if source.startswith(prefix):
            return side, source[len(prefix):]
    return None, source


def analysis_topk_pathway(outdir: Path) -> pd.DataFrame:
    print("\n=== Top-k pathway recovery ===")
    ks = [5, 10, 20, 50]
    all_rows = []

    for csv_name, level_label in [
        ("gsea_level2_results.csv", "L2"),
        ("gsea_level3_results.csv", "L3"),
    ]:
        csv_path = GSEA_DIR / csv_name
        if not csv_path.exists():
            print(f"  {csv_name} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        df[["side", "key"]] = df["source"].apply(lambda s: pd.Series(_parse_source(s)))
        df = df[df["side"].notna()]

        for key in sorted(df["key"].unique()):
            real_sub = df[(df["key"] == key) & (df["side"] == "real")].copy()
            pred_sub = df[(df["key"] == key) & (df["side"] == "pred")].copy()
            if real_sub.empty or pred_sub.empty:
                continue

            real_ranked = real_sub.sort_values("NES", key=abs, ascending=False)["Term"].tolist()
            pred_ranked = pred_sub.sort_values("NES", key=abs, ascending=False)["Term"].tolist()

            row = {"key": key, "level": level_label}
            for k in ks:
                real_topk = set(real_ranked[:k])
                pred_topk = set(pred_ranked[:k])
                if len(real_topk) == 0:
                    row[f"precision@{k}"] = np.nan
                else:
                    row[f"precision@{k}"] = len(real_topk & pred_topk) / len(real_topk)
            all_rows.append(row)

    if not all_rows:
        print("  No conditions computed.")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    for level_label in result["level"].unique():
        sub = result[result["level"] == level_label]
        prec_cols = [c for c in sub.columns if c.startswith("precision@")]

        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in sub.iterrows():
            vals = [row[c] for c in prec_cols]
            ax.plot(ks[:len(vals)], vals, "o-", alpha=0.5, markersize=4, label=row["key"])

        mean_vals = [sub[c].mean() for c in prec_cols]
        ax.plot(ks[:len(mean_vals)], mean_vals, "s-", color="black", lw=2.5,
                markersize=7, label="Mean", zorder=10)

        ax.set_xlabel("k (top-k pathways)")
        ax.set_ylabel("Precision@k")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Top-k pathway recovery ({level_label})")
        if len(sub) <= 12:
            ax.legend(fontsize=6, loc="best", ncol=2)
        plt.tight_layout()
        fig.savefig(outdir / f"topk_pathway_{level_label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved topk_pathway_{level_label}.png")

    agg_rows = []
    for level_label in result["level"].unique():
        sub = result[result["level"] == level_label]
        prec_cols = [c for c in sub.columns if c.startswith("precision@")]
        agg = {"level": level_label}
        for c in prec_cols:
            agg[f"mean_{c}"] = sub[c].mean()
            agg[f"median_{c}"] = sub[c].median()
        agg_rows.append(agg)
    agg_df = pd.DataFrame(agg_rows)
    print("\n  Aggregated top-k recovery:")
    print(agg_df.to_string(index=False))

    return result


# ---------------------------------------------------------------------------
# 4. Distribution-level metric
# ---------------------------------------------------------------------------

def analysis_distribution(outdir: Path) -> pd.DataFrame:
    print("\n=== Distribution-level metric (Wasserstein / KS) ===")
    rows = []

    for ds_name in EXTERNAL_DATASETS:
        adata_real, adata_pred, ds_info = _load_external_pair(ds_name)
        cyto = ds_info["atlas_cytokine"]
        pert_col = ds_info["pert_col"]
        ctrl = ds_info["control_label"]

        cell_types = sorted(adata_real.obs["cell_type"].unique())
        per_ct_wasserstein = {}

        for ct in cell_types:
            trt_real = (adata_real.obs["cell_type"] == ct) & (adata_real.obs[pert_col] != ctrl)
            trt_pred = (adata_pred.obs["cell_type"] == ct) & (adata_pred.obs[pert_col] != ctrl)

            n_real, n_pred = int(trt_real.sum()), int(trt_pred.sum())
            if min(n_real, n_pred) < 3:
                continue

            X_real = _to_dense(adata_real[trt_real].X)
            X_pred = _to_dense(adata_pred[trt_pred].X)
            n_genes = X_real.shape[1]

            wass = np.zeros(n_genes)
            ks_stats = np.zeros(n_genes)
            for g in range(n_genes):
                r, p = X_real[:, g], X_pred[:, g]
                wass[g] = wasserstein_distance(r, p)
                ks_stats[g] = ks_2samp(r, p).statistic

            per_ct_wasserstein[(ds_name, ct)] = wass

            rows.append({
                "dataset": ds_name, "cell_type": ct, "cytokine": cyto,
                "wasserstein_median": float(np.median(wass)),
                "wasserstein_mean": float(np.mean(wass)),
                "ks_median": float(np.median(ks_stats)),
                "ks_mean": float(np.mean(ks_stats)),
                "n_real": n_real, "n_pred": n_pred, "n_genes": n_genes,
            })
            print(f"  {ds_name} {ct}: median_W={np.median(wass):.4f}, median_KS={np.median(ks_stats):.3f}")

        if per_ct_wasserstein:
            labels = []
            data = []
            for (ds, ct), w in sorted(per_ct_wasserstein.items()):
                labels.append(f"{ct}\n({ds})")
                data.append(w)

            fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.8), 5))
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.5))
            for patch in bp["boxes"]:
                patch.set_facecolor("#92c5de")
                patch.set_alpha(0.7)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Wasserstein distance (per gene)")
            ax.set_title(f"Distribution-level: per-gene Wasserstein distance — {ds_name}")
            plt.tight_layout()
            fig.savefig(outdir / f"distribution_wasserstein_box_{ds_name}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved distribution_wasserstein_box_{ds_name}.png")

    if not rows:
        print("  No conditions computed.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.6), 5))
    labels = [f"{r['cell_type']}\n({r['dataset']})" for _, r in df.iterrows()]
    ax.bar(range(len(df)), df["wasserstein_median"].values, color="#fdb863",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Median Wasserstein distance")
    ax.set_title("Distribution-level metric: median per-gene Wasserstein distance")
    plt.tight_layout()
    fig.savefig(outdir / "distribution_wasserstein_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved distribution_wasserstein_summary.png")

    return df


# ---------------------------------------------------------------------------
# 5. Failure-mode stratification
# ---------------------------------------------------------------------------

def analysis_failure_modes(outdir: Path) -> None:
    print("\n=== Failure-mode stratification ===")

    coherence = pd.read_csv(COHERENCE_PATH) if COHERENCE_PATH.exists() else pd.DataFrame()
    pseudobulk = pd.read_csv(PSEUDOBULK_PATH) if PSEUDOBULK_PATH.exists() else pd.DataFrame()
    mean_expr = pd.read_csv(MEAN_EXPR_PATH) if MEAN_EXPR_PATH.exists() else pd.DataFrame()

    if coherence.empty:
        print("  gene_coherence_metrics.csv not found, skipping")
        return

    join_cols = ["dataset", "cell_type", "cytokine"]

    master = coherence.copy()
    if not pseudobulk.empty:
        master = master.merge(pseudobulk[join_cols + ["r_delta_pearson", "n_ctrl", "n_trt"]],
                              on=join_cols, how="left")
    if not mean_expr.empty:
        master = master.merge(mean_expr[join_cols + ["pearson_mean", "n_cells"]],
                              on=join_cols, how="left")

    if "n_trt" not in master.columns:
        master["n_trt"] = np.nan
    if "n_ctrl" not in master.columns:
        master["n_ctrl"] = np.nan
    master["total_cells"] = master[["n_ctrl", "n_trt"]].sum(axis=1, min_count=1)

    metric_pairs = [
        ("spearman_logFC", "Spearman logFC"),
        ("jaccard_de", "Jaccard DE"),
        ("pearson_mean", "Pearson mean expr"),
    ]
    covariate_pairs = [
        ("total_cells", "Total cells (ctrl + treated)"),
        ("n_de_atlas", "Number of DE genes (atlas)"),
    ]

    for metric_col, metric_label in metric_pairs:
        if metric_col not in master.columns:
            continue
        for cov_col, cov_label in covariate_pairs:
            if cov_col not in master.columns:
                continue
            valid = master[[metric_col, cov_col]].dropna()
            if len(valid) < 5:
                continue

            x = valid[cov_col].values
            y = valid[metric_col].values
            r_s, p_s = spearmanr(x, y)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(x, y, s=40, alpha=0.7, c="steelblue", edgecolors="white", linewidths=0.5)

            for i, row in master.iterrows():
                if pd.notna(row.get(metric_col)) and pd.notna(row.get(cov_col)):
                    ax.annotate(row["cell_type"], (row[cov_col], row[metric_col]),
                                fontsize=5, alpha=0.7, ha="center", va="bottom")

            ax.set_xlabel(cov_label)
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} vs {cov_label}\n"
                         f"Spearman r = {r_s:.3f} (p = {p_s:.3g})", fontsize=10)
            plt.tight_layout()
            safe = f"{metric_col}_vs_{cov_col}".replace(" ", "_")
            fig.savefig(outdir / f"failure_{safe}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved failure_{safe}.png  (r_s={r_s:.3f})")

    ct_agg = master.groupby("cell_type").agg(
        spearman_logFC_mean=("spearman_logFC", "mean"),
        jaccard_de_mean=("jaccard_de", "mean"),
        n_conditions=("dataset", "count"),
    ).reset_index()

    if len(ct_agg) >= 3:
        metrics_to_plot = ["spearman_logFC_mean", "jaccard_de_mean"]
        fig, axes = plt.subplots(1, len(metrics_to_plot),
                                 figsize=(6 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics_to_plot):
            ct_sorted = ct_agg.sort_values(metric, ascending=True)
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ct_sorted)))
            ax.barh(range(len(ct_sorted)), ct_sorted[metric].values,
                    color=colors, edgecolor="black", linewidth=0.5)
            ax.set_yticks(range(len(ct_sorted)))
            ax.set_yticklabels(ct_sorted["cell_type"], fontsize=8)
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()}\n(mean across datasets)")

        plt.suptitle("Failure-mode: performance by cell type", fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(outdir / "failure_celltype_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved failure_celltype_bar.png")

    print("  Failure-mode stratification complete.")


# ---------------------------------------------------------------------------
# 6. Master summary heatmap
# ---------------------------------------------------------------------------

def _compute_nes_pearson() -> pd.DataFrame:
    """Compute per-condition NES Pearson r from GSEA level 3 results."""
    csv_path = GSEA_DIR / "gsea_level3_results.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df[["side", "key"]] = df["source"].apply(lambda s: pd.Series(_parse_source(s)))
    df = df[df["side"].notna()]

    real_df = df[df["side"] == "real"][["Term", "NES", "key"]].rename(columns={"NES": "NES_real"})
    pred_df = df[df["side"] == "pred"][["Term", "NES", "key"]].rename(columns={"NES": "NES_pred"})
    merged = real_df.merge(pred_df, on=["Term", "key"], how="inner")

    rows = []
    for key in merged["key"].unique():
        sub = merged[merged["key"] == key]
        x = sub["NES_real"].values
        y = sub["NES_pred"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        r_p = pearsonr(x[mask], y[mask])[0]

        m = re.match(r"(GSE\d+)_(.*?)_(IFN-\w+)", key)
        if m:
            ds, ct, cyto = m.group(1), m.group(2).replace("_", " "), m.group(3)
        else:
            continue
        rows.append({"dataset": ds, "cell_type": ct, "cytokine": cyto, "nes_pearson": r_p})

    return pd.DataFrame(rows)


def analysis_master_heatmap(outdir: Path, cal_df: pd.DataFrame | None = None,
                            dir_df: pd.DataFrame | None = None) -> None:
    print("\n=== Master summary heatmap ===")

    join_cols = ["dataset", "cell_type", "cytokine"]
    coherence = pd.read_csv(COHERENCE_PATH) if COHERENCE_PATH.exists() else pd.DataFrame()
    mean_expr = pd.read_csv(MEAN_EXPR_PATH) if MEAN_EXPR_PATH.exists() else pd.DataFrame()
    pseudobulk = pd.read_csv(PSEUDOBULK_PATH) if PSEUDOBULK_PATH.exists() else pd.DataFrame()

    if coherence.empty:
        print("  gene_coherence_metrics.csv not found, aborting")
        return

    master = coherence[join_cols + ["spearman_logFC", "jaccard_de", "precision_de", "recall_de"]].copy()

    if not mean_expr.empty:
        master = master.merge(
            mean_expr[join_cols + ["pearson_mean", "spearman_mean"]],
            on=join_cols, how="left",
        )
    if not pseudobulk.empty:
        master = master.merge(
            pseudobulk[join_cols + ["r_delta_pearson", "r_delta_spearman"]],
            on=join_cols, how="left",
        )

    nes_df = _compute_nes_pearson()
    if not nes_df.empty:
        master = master.merge(nes_df, on=join_cols, how="left")

    if cal_df is not None and not cal_df.empty:
        master = master.merge(
            cal_df[join_cols + ["slope"]].rename(columns={"slope": "calibration_slope"}),
            on=join_cols, how="left",
        )
    if dir_df is not None and not dir_df.empty:
        master = master.merge(
            dir_df[join_cols + ["dir_acc_all"]].rename(columns={"dir_acc_all": "direction_accuracy"}),
            on=join_cols, how="left",
        )

    master["label"] = master["cell_type"] + " (" + master["dataset"] + ")"
    master = master.set_index("label")

    metric_cols = [c for c in master.columns if c not in join_cols + ["label"]
                   and master[c].dtype in (np.float64, float)]
    metric_cols = [c for c in metric_cols if master[c].notna().sum() >= 3]

    if not metric_cols:
        print("  No metric columns with enough data, aborting")
        return

    plot_data = master[metric_cols].copy()

    display_names = {
        "spearman_logFC": "Spearman logFC",
        "jaccard_de": "Jaccard DE",
        "precision_de": "Precision DE",
        "recall_de": "Recall DE",
        "pearson_mean": "Pearson mean expr",
        "spearman_mean": "Spearman mean expr",
        "r_delta_pearson": "Pseudobulk delta r",
        "r_delta_spearman": "Pseudobulk delta rho",
        "nes_pearson": "NES Pearson r",
        "calibration_slope": "Calibration slope",
        "direction_accuracy": "Direction accuracy",
    }
    plot_data.columns = [display_names.get(c, c) for c in plot_data.columns]

    annot_data = plot_data.copy()

    zscore_data = plot_data.copy()
    for col in zscore_data.columns:
        vals = zscore_data[col]
        mu, sigma = vals.mean(), vals.std()
        if sigma > 0:
            zscore_data[col] = (vals - mu) / sigma
        else:
            zscore_data[col] = 0.0

    fig, ax = plt.subplots(figsize=(max(12, len(metric_cols) * 1.2), max(6, len(plot_data) * 0.45)))
    sns.heatmap(
        zscore_data, ax=ax,
        annot=annot_data.round(3).values,
        fmt="",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Z-score", "shrink": 0.7},
        xticklabels=plot_data.columns,
        yticklabels=plot_data.index,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title("Master summary: all validation metrics\n(Z-scored colour, raw values annotated)", fontsize=11)
    plt.tight_layout()
    fig.savefig(outdir / "master_summary_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved master_summary_heatmap.png ({plot_data.shape[0]} conditions x {plot_data.shape[1]} metrics)")

    csv_path = outdir / "master_summary_table.csv"
    annot_data.to_csv(csv_path)
    print(f"  Saved master_summary_table.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 9: Extended validation analyses")
    parser.add_argument(
        "--analysis",
        choices=["calibration", "direction", "topk_pathway", "distribution",
                 "failure_modes", "master_heatmap"],
        default=None,
        help="Run a single analysis (default: all)",
    )
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    run_all = args.analysis is None

    if run_all or args.analysis == "calibration":
        cal_df = analysis_calibration(OUTDIR)
        all_metrics["calibration"] = cal_df

    if run_all or args.analysis == "direction":
        dir_df = analysis_direction(OUTDIR)
        all_metrics["direction"] = dir_df

    if run_all or args.analysis == "topk_pathway":
        topk_df = analysis_topk_pathway(OUTDIR)
        all_metrics["topk_pathway"] = topk_df

    if run_all or args.analysis == "distribution":
        dist_df = analysis_distribution(OUTDIR)
        all_metrics["distribution"] = dist_df

    if run_all or args.analysis == "failure_modes":
        analysis_failure_modes(OUTDIR)

    if run_all or args.analysis == "master_heatmap":
        cal_df = all_metrics.get("calibration")
        dir_df = all_metrics.get("direction")
        analysis_master_heatmap(OUTDIR, cal_df=cal_df, dir_df=dir_df)

    if all_metrics:
        summary_parts = []
        for name, df in all_metrics.items():
            if df is not None and not df.empty:
                df = df.copy()
                df.insert(0, "analysis", name)
                summary_parts.append(df)
        if summary_parts:
            combined = pd.concat(summary_parts, ignore_index=True)
            csv_path = BIO_DIR / "extended_validation_metrics.csv"
            combined.to_csv(csv_path, index=False)
            print(f"\nSaved combined metrics to {csv_path.name} ({len(combined)} rows)")

    print("\nStep 9 (extended validation) complete.")


if __name__ == "__main__":
    main()
