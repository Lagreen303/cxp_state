#!/usr/bin/env python3
"""
Biological coherence analyses for cytokine validation.

Computes gene-level DE coherence between the cytokine atlas and external
predictions, generates volcano plots, logFC scatter plots, pred-vs-real
expression scatters, and per-dataset UMAPs.

All analyses on prediction-derived biology use non-imputed genes only.

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/run_biological_coherence.py
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
RUN = BASE / "run_2026_02_11_all"
OUTDIR = BASE / "biological_coherence"
OUTDIR.mkdir(parents=True, exist_ok=True)

ATLAS_PATH = Path("/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad")

EXTERNAL_DATASETS = {
    "GSE96583": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/GSE96583_batch2_IFNbeta_aligned_avg.h5ad"),
        "pred_h5ad": RUN / "predictions/GSE96583_batch2_IFNbeta_aligned_avg_predictions.h5ad",
        "clustered_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/clustering/GSE96583_batch2_IFNbeta_clustered_manual_annotated.h5ad"),
        "eval_no_imputed": RUN / "GSE96583_eval/results_eval_GSE96583_batch2_IFNbeta_aligned_avg_predictions_no_imputed_genes",
        "perturbation": "IFN-beta",
        "pert_col": "cytokine",
        "control_label": "control",
    },
    "GSE178429": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg.h5ad"),
        "pred_h5ad": RUN / "predictions/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_predictions.h5ad",
        "clustered_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/clustering/GSE178429_PBMCs_stim_scRNAseq_clustered_manual_annotated.h5ad"),
        "eval_no_imputed": RUN / "GSE178429_eval/results_eval_GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_predictions_no_imputed_genes",
        "perturbation": "IFN-gamma",
        "pert_col": "cytokine",
        "control_label": "control",
    },
}

ATLAS_PERT_COL = "cytokine"
ATLAS_CELLTYPE_COL = "cell_type"
ATLAS_CONTROL = "PBS"

FDR_THRESH = 0.05
GSEA_FDR_THRESH = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_log2fc(fc: pd.Series) -> pd.Series:
    """Convert fold_change to log2FC, handling zeros/negatives."""
    fc = fc.copy()
    fc[fc <= 0] = np.nan
    return np.log2(fc)


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def load_de_table(path: Path) -> pd.DataFrame:
    """Load a cell-eval style DE CSV (target, reference, feature, ..., fdr)."""
    df = pd.read_csv(path)
    df["log2FC"] = _safe_log2fc(df["fold_change"])
    return df


def collect_cell_types_from_eval(eval_dir: Path) -> list[str]:
    """Extract cell type names from the *_real_de.csv files in an eval dir."""
    cts = []
    for f in eval_dir.glob("*_real_de.csv"):
        ct = f.name.replace("_real_de.csv", "")
        cts.append(ct)
    return sorted(cts)


# ===========================================================================
# STEP 1: Atlas DE
# ===========================================================================

def compute_atlas_de(
    atlas_path: Path,
    output_path: Path,
    pert_col: str = ATLAS_PERT_COL,
    celltype_col: str = ATLAS_CELLTYPE_COL,
    control: str = ATLAS_CONTROL,
) -> pd.DataFrame:
    """Compute DE for each (cytokine, cell_type) in the atlas using Wilcoxon."""

    if output_path.exists():
        print(f"[atlas DE] Loading cached results from {output_path}")
        return pd.read_csv(output_path)

    print(f"[atlas DE] Loading atlas from {atlas_path} ...")
    adata = ad.read_h5ad(atlas_path)
    print(f"  shape: {adata.shape}")

    if issparse(adata.X):
        adata.X = adata.X.tocsr()

    # Normalize + log1p so Wilcoxon operates on log-space data
    X_sample = _to_dense(adata.X[:100])
    is_counts = (X_sample >= 0).all() and np.allclose(X_sample, X_sample.astype(int))
    if is_counts:
        print("  Normalising atlas (counts detected) ...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    all_rows = []
    cell_types = adata.obs[celltype_col].unique()
    cytokines = [c for c in adata.obs[pert_col].unique() if c != control]
    print(f"  {len(cell_types)} cell types, {len(cytokines)} cytokines (excl. {control})")

    for ct in sorted(cell_types):
        ct_mask = adata.obs[celltype_col] == ct
        ct_ctrl_mask = ct_mask & (adata.obs[pert_col] == control)
        n_ctrl = ct_ctrl_mask.sum()
        if n_ctrl < 3:
            continue

        ct_count = 0
        for cyto in sorted(cytokines):
            pert_mask = ct_mask & (adata.obs[pert_col] == cyto)
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
        print("[atlas DE] No DE results computed!")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df.rename(columns={"names": "gene", "logfoldchanges": "logFC_atlas",
                            "pvals": "pval_atlas", "pvals_adj": "fdr_atlas",
                            "scores": "score_atlas"})
    df.to_csv(output_path, index=False)
    print(f"[atlas DE] Saved {len(df)} rows to {output_path}")
    return df


# ===========================================================================
# STEP 2: Gene-level coherence metrics
# ===========================================================================

@dataclass
class CoherenceResult:
    dataset: str
    cell_type: str
    cytokine: str
    variant: str
    spearman_logFC: float = np.nan
    pearson_logFC: float = np.nan
    jaccard_de: float = np.nan
    precision_de: float = np.nan
    recall_de: float = np.nan
    n_de_atlas: int = 0
    n_de_pred: int = 0
    n_de_real_ext: int = 0
    n_shared_genes: int = 0


def compute_gene_coherence(
    atlas_de: pd.DataFrame,
    ext_eval_dir: Path,
    dataset_name: str,
    variant: str = "avg_no_imputed",
) -> list[CoherenceResult]:
    """Compute gene-level coherence between atlas DE and external pred DE."""

    results = []
    cell_types = collect_cell_types_from_eval(ext_eval_dir)

    for ct in cell_types:
        pred_de_path = ext_eval_dir / f"{ct}_pred_de.csv"
        real_de_path = ext_eval_dir / f"{ct}_real_de.csv"

        if not pred_de_path.exists():
            continue

        pred_de = load_de_table(pred_de_path)
        real_de = load_de_table(real_de_path) if real_de_path.exists() else None

        cytokines_in_pred = pred_de["target"].unique()

        for cyto in cytokines_in_pred:
            pred_cyto = pred_de[pred_de["target"] == cyto].copy()

            # Find matching atlas DE
            # Map external perturbation names to atlas names
            atlas_cyto_name = cyto
            atlas_ct = atlas_de[(atlas_de["cytokine"] == atlas_cyto_name) &
                                (atlas_de["cell_type"] == ct)]

            if atlas_ct.empty:
                # Try fuzzy matching (IFN-gamma -> IFNg etc.)
                atlas_cyto_candidates = atlas_de["cytokine"].unique()
                matched = [c for c in atlas_cyto_candidates
                           if cyto.lower().replace("-", "").replace(" ", "") in
                           c.lower().replace("-", "").replace(" ", "")]
                if matched:
                    atlas_cyto_name = matched[0]
                    atlas_ct = atlas_de[(atlas_de["cytokine"] == atlas_cyto_name) &
                                        (atlas_de["cell_type"] == ct)]

            if atlas_ct.empty:
                continue

            merged = pd.merge(
                atlas_ct[["gene", "logFC_atlas", "fdr_atlas"]],
                pred_cyto[["feature", "log2FC", "fdr"]].rename(
                    columns={"feature": "gene", "log2FC": "logFC_pred", "fdr": "fdr_pred"}),
                on="gene", how="inner"
            )
            merged = merged.dropna(subset=["logFC_atlas", "logFC_pred"])

            if len(merged) < 10:
                continue

            r = CoherenceResult(
                dataset=dataset_name, cell_type=ct, cytokine=cyto, variant=variant,
                n_shared_genes=len(merged),
            )

            # LogFC correlations
            try:
                r.spearman_logFC = spearmanr(merged["logFC_atlas"], merged["logFC_pred"])[0]
                r.pearson_logFC = pearsonr(merged["logFC_atlas"], merged["logFC_pred"])[0]
            except Exception:
                pass

            # DE set overlap
            atlas_sig = set(merged.loc[merged["fdr_atlas"] < FDR_THRESH, "gene"])
            pred_sig = set(merged.loc[merged["fdr_pred"] < FDR_THRESH, "gene"])
            r.n_de_atlas = len(atlas_sig)
            r.n_de_pred = len(pred_sig)

            if real_de is not None:
                real_cyto = real_de[real_de["target"] == cyto]
                r.n_de_real_ext = (real_cyto["fdr"] < FDR_THRESH).sum()

            union = atlas_sig | pred_sig
            inter = atlas_sig & pred_sig
            r.jaccard_de = len(inter) / len(union) if union else np.nan
            r.precision_de = len(inter) / len(pred_sig) if pred_sig else np.nan
            r.recall_de = len(inter) / len(atlas_sig) if atlas_sig else np.nan

            results.append(r)

    return results


# ===========================================================================
# STEP 3: Volcano plots, logFC scatter, heatmaps
# ===========================================================================

def make_volcano_plot(
    de_df: pd.DataFrame,
    atlas_de_genes: set | None,
    cell_type: str,
    cytokine: str,
    dataset_name: str,
    outdir: Path,
):
    """Volcano plot from pred_de, highlighting atlas-confirmed DE genes."""
    df = de_df.copy()
    df["neg_log10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))
    df["log2FC"] = _safe_log2fc(df["fold_change"])
    df = df.dropna(subset=["log2FC", "neg_log10_fdr"])

    fig, ax = plt.subplots(figsize=(7, 5))

    sig_mask = df["fdr"] < FDR_THRESH
    nonsig = df[~sig_mask]
    sig = df[sig_mask]

    ax.scatter(nonsig["log2FC"], nonsig["neg_log10_fdr"], s=4, alpha=0.3, c="grey", label="Not sig")
    ax.scatter(sig["log2FC"], sig["neg_log10_fdr"], s=6, alpha=0.5, c="steelblue", label=f"Pred DE (FDR<{FDR_THRESH})")

    if atlas_de_genes:
        in_atlas = sig[sig["feature"].isin(atlas_de_genes)]
        ax.scatter(in_atlas["log2FC"], in_atlas["neg_log10_fdr"], s=12, alpha=0.8,
                   c="firebrick", label=f"Also DE in atlas ({len(in_atlas)})", zorder=5)

        # Label top atlas-confirmed genes
        top = in_atlas.nlargest(8, "neg_log10_fdr")
        for _, row in top.iterrows():
            ax.annotate(row["feature"], (row["log2FC"], row["neg_log10_fdr"]),
                        fontsize=6, ha="center", va="bottom")

    ax.axhline(-np.log10(FDR_THRESH), ls="--", c="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("log2(fold change)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Volcano: {dataset_name} {cell_type} - {cytokine}\n(predicted DE, non-imputed genes)")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()

    ct_safe = cell_type.replace(" ", "_").replace("/", "-")
    fname = outdir / f"volcano_{dataset_name}_{ct_safe}_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def make_logfc_scatter(
    merged: pd.DataFrame,
    cell_type: str,
    cytokine: str,
    dataset_name: str,
    outdir: Path,
):
    """Scatter of logFC_atlas vs logFC_pred, colored by DE category."""
    df = merged.dropna(subset=["logFC_atlas", "logFC_pred"]).copy()
    if len(df) < 5:
        return None

    atlas_sig = df["fdr_atlas"] < FDR_THRESH
    pred_sig = df["fdr_pred"] < FDR_THRESH

    df["category"] = "Non-DE"
    df.loc[atlas_sig & ~pred_sig, "category"] = "Atlas only"
    df.loc[~atlas_sig & pred_sig, "category"] = "Pred only"
    df.loc[atlas_sig & pred_sig, "category"] = "Both"

    colors = {"Non-DE": "lightgrey", "Atlas only": "royalblue", "Pred only": "orange", "Both": "firebrick"}

    r_sp = spearmanr(df["logFC_atlas"], df["logFC_pred"])[0]
    r_pe = pearsonr(df["logFC_atlas"], df["logFC_pred"])[0]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for cat in ["Non-DE", "Atlas only", "Pred only", "Both"]:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub["logFC_atlas"], sub["logFC_pred"], s=6, alpha=0.5,
                   c=colors[cat], label=f"{cat} ({len(sub)})")

    lims = [min(df["logFC_atlas"].min(), df["logFC_pred"].min()),
            max(df["logFC_atlas"].max(), df["logFC_pred"].max())]
    ax.plot(lims, lims, "--", c="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("logFC (atlas)")
    ax.set_ylabel("logFC (predicted external)")
    ax.set_title(f"{dataset_name} {cell_type} - {cytokine}\nSpearman={r_sp:.2f}, Pearson={r_pe:.2f}")
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()

    ct_safe = cell_type.replace(" ", "_").replace("/", "-")
    fname = outdir / f"scatter_logFC_{dataset_name}_{ct_safe}_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def make_coherence_heatmap(
    coherence_df: pd.DataFrame,
    metric: str,
    outdir: Path,
    dataset_name: str | None = None,
):
    """Heatmap of a coherence metric: cytokines x cell_types."""
    df = coherence_df.copy()
    if dataset_name:
        df = df[df["dataset"] == dataset_name]

    if df.empty:
        return None

    pivot = df.pivot_table(index="cytokine", columns="cell_type", values=metric, aggfunc="mean")
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.8),
                                     max(4, len(pivot.index) * 0.4)))
    import seaborn as sns
    cmap = "RdYlGn" if "jaccard" not in metric else "YlOrRd"
    vmin = -1 if "correlation" in metric or "spearman" in metric or "pearson" in metric else 0
    vmax = 1

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    title = f"{metric}"
    if dataset_name:
        title = f"{dataset_name}: {metric}"
    ax.set_title(title)
    plt.tight_layout()

    ds_tag = dataset_name or "all"
    fname = outdir / f"heatmap_{metric}_{ds_tag}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


# ===========================================================================
# STEP 4: Pred vs real expression scatters
# ===========================================================================

def make_pred_vs_real_scatter(
    adata_real: ad.AnnData,
    adata_pred: ad.AnnData,
    cell_type: str,
    cytokine: str,
    dataset_name: str,
    pert_col: str,
    celltype_col: str,
    outdir: Path,
):
    """Mean expression scatter: real vs pred for treated cells (non-imputed)."""
    mask_real = (adata_real.obs[celltype_col] == cell_type) & (adata_real.obs[pert_col] == cytokine)
    mask_pred = (adata_pred.obs[celltype_col] == cell_type) & (adata_pred.obs[pert_col] == cytokine)

    n_real = mask_real.sum()
    n_pred = mask_pred.sum()
    if n_real < 3 or n_pred < 3:
        return None

    X_real = _to_dense(adata_real[mask_real].X)
    X_pred = _to_dense(adata_pred[mask_pred].X)

    mean_real = X_real.mean(axis=0)
    mean_pred = X_pred.mean(axis=0)

    r = pearsonr(mean_real, mean_pred)[0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mean_real, mean_pred, s=3, alpha=0.3, c="steelblue")
    lims = [0, max(mean_real.max(), mean_pred.max()) * 1.05]
    ax.plot(lims, lims, "--", c="red", lw=0.8, alpha=0.6)
    ax.set_xlabel(f"Mean expression (real, n={n_real})")
    ax.set_ylabel(f"Mean expression (predicted, n={n_pred})")
    ax.set_title(f"{dataset_name} {cell_type} - {cytokine}\nPearson r = {r:.3f}")
    plt.tight_layout()

    ct_safe = cell_type.replace(" ", "_").replace("/", "-")
    fname = outdir / f"pred_vs_real_{dataset_name}_{ct_safe}_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


# ===========================================================================
# STEP 5: UMAPs
# ===========================================================================

def make_umap_for_dataset(
    adata: ad.AnnData,
    dataset_name: str,
    celltype_col: str,
    pert_col: str,
    outdir: Path,
    source_col: str | None = None,
    max_cells: int = 50000,
    n_top_genes: int = 2000,
):
    """Run UMAP and save plots colored by cell type, perturbation, and source."""
    adata = adata.copy()

    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells)

    X_sample = _to_dense(adata.X[:min(200, adata.n_obs)])
    is_counts = (X_sample >= 0).all() and np.allclose(X_sample, X_sample.astype(int))
    if is_counts:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars), subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata, min_dist=0.3)

    ds_safe = dataset_name.replace(" ", "_")

    color_cols = []
    if celltype_col in adata.obs.columns:
        color_cols.append(celltype_col)
    if pert_col in adata.obs.columns:
        color_cols.append(pert_col)
    if source_col and source_col in adata.obs.columns:
        color_cols.append(source_col)

    for col in color_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata, color=col, ax=ax, show=False, size=8, alpha=0.6,
                    legend_fontsize=6, frameon=True)
        ax.set_title(f"{dataset_name} - {col}")
        plt.tight_layout()
        fname = outdir / f"umap_{ds_safe}_{col.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)

    return adata


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    import seaborn as sns
    sns.set_style("whitegrid")

    plots_dir = OUTDIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    volcano_dir = plots_dir / "volcanos"
    volcano_dir.mkdir(exist_ok=True)
    scatter_dir = plots_dir / "logfc_scatters"
    scatter_dir.mkdir(exist_ok=True)
    expr_dir = plots_dir / "pred_vs_real_expression"
    expr_dir.mkdir(exist_ok=True)
    umap_dir = plots_dir / "umaps"
    umap_dir.mkdir(exist_ok=True)
    heatmap_dir = plots_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1: Atlas DE
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Computing atlas DE per (cytokine, cell_type)")
    print("=" * 60)
    atlas_de_path = OUTDIR / "atlas_de_results.csv"
    atlas_de = compute_atlas_de(ATLAS_PATH, atlas_de_path)

    if atlas_de.empty:
        print("ERROR: No atlas DE results. Cannot proceed.")
        return

    print(f"Atlas DE: {len(atlas_de)} rows, "
          f"{atlas_de['cytokine'].nunique()} cytokines, "
          f"{atlas_de['cell_type'].nunique()} cell types")

    # -----------------------------------------------------------------------
    # STEP 2: Gene-level coherence metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Computing gene-level coherence metrics")
    print("=" * 60)

    all_coherence = []
    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        eval_dir = ds_info["eval_no_imputed"]
        if not eval_dir.exists():
            print(f"  [SKIP] {ds_name}: eval dir not found: {eval_dir}")
            continue

        print(f"\n  Processing {ds_name} ...")
        results = compute_gene_coherence(atlas_de, eval_dir, ds_name)
        all_coherence.extend(results)
        print(f"    {len(results)} coherence results computed")

    if all_coherence:
        coherence_df = pd.DataFrame([vars(r) for r in all_coherence])
        coherence_path = OUTDIR / "gene_coherence_metrics.csv"
        coherence_df.to_csv(coherence_path, index=False)
        print(f"\nSaved gene coherence metrics to {coherence_path}")
        print(coherence_df[["dataset", "cell_type", "cytokine", "spearman_logFC",
                            "jaccard_de", "precision_de", "recall_de"]].to_string(index=False))
    else:
        coherence_df = pd.DataFrame()
        print("WARNING: No coherence results computed.")

    # -----------------------------------------------------------------------
    # STEP 3: Volcano plots, logFC scatters, heatmaps
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Generating volcano plots and logFC scatters")
    print("=" * 60)

    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        eval_dir = ds_info["eval_no_imputed"]
        if not eval_dir.exists():
            continue

        cell_types = collect_cell_types_from_eval(eval_dir)
        for ct in cell_types:
            pred_de_path = eval_dir / f"{ct}_pred_de.csv"
            if not pred_de_path.exists():
                continue

            pred_de = load_de_table(pred_de_path)

            for cyto in pred_de["target"].unique():
                pred_cyto = pred_de[pred_de["target"] == cyto]

                # Atlas DE genes for this (cytokine, cell_type)
                atlas_cyto_name = cyto
                atlas_ct = atlas_de[(atlas_de["cytokine"] == atlas_cyto_name) &
                                    (atlas_de["cell_type"] == ct)]
                if atlas_ct.empty:
                    candidates = atlas_de["cytokine"].unique()
                    matched = [c for c in candidates
                               if cyto.lower().replace("-", "").replace(" ", "") in
                               c.lower().replace("-", "").replace(" ", "")]
                    if matched:
                        atlas_cyto_name = matched[0]
                        atlas_ct = atlas_de[(atlas_de["cytokine"] == atlas_cyto_name) &
                                            (atlas_de["cell_type"] == ct)]

                atlas_sig_genes = set(atlas_ct.loc[atlas_ct["fdr_atlas"] < FDR_THRESH, "gene"]) if not atlas_ct.empty else None

                # Volcano
                make_volcano_plot(pred_cyto, atlas_sig_genes, ct, cyto, ds_name, volcano_dir)

                # LogFC scatter
                if not atlas_ct.empty:
                    merged = pd.merge(
                        atlas_ct[["gene", "logFC_atlas", "fdr_atlas"]],
                        pred_cyto[["feature", "log2FC", "fdr"]].rename(
                            columns={"feature": "gene", "log2FC": "logFC_pred", "fdr": "fdr_pred"}),
                        on="gene", how="inner"
                    )
                    make_logfc_scatter(merged, ct, cyto, ds_name, scatter_dir)

                print(f"  {ds_name}/{ct}/{cyto}: volcano + scatter done")

    # Heatmaps
    if not coherence_df.empty:
        print("\n  Generating coherence heatmaps ...")
        for ds_name in coherence_df["dataset"].unique():
            for metric in ["spearman_logFC", "jaccard_de", "precision_de", "recall_de"]:
                make_coherence_heatmap(coherence_df, metric, heatmap_dir, ds_name)
        print("  Heatmaps done.")

    # -----------------------------------------------------------------------
    # STEP 4: Pred vs real expression scatters
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Generating pred vs real expression scatters")
    print("=" * 60)

    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        real_path = ds_info["real_h5ad"]
        pred_path = ds_info["pred_h5ad"]

        if not real_path.exists() or not pred_path.exists():
            print(f"  [SKIP] {ds_name}: missing h5ad")
            continue

        print(f"  Loading {ds_name} real+pred h5ad ...")
        adata_real = ad.read_h5ad(real_path)
        adata_pred = ad.read_h5ad(pred_path)

        # Subset to non-imputed genes
        if "is_imputed" in adata_real.var.columns:
            keep = ~adata_real.var["is_imputed"].astype(bool).values
            adata_real = adata_real[:, keep].copy()
            adata_pred = adata_pred[:, keep].copy()
            print(f"    Kept {keep.sum()} non-imputed genes out of {len(keep)}")

        pert_col = ds_info["pert_col"]
        cyto = ds_info["perturbation"]
        celltype_col = "cell_type"

        if celltype_col not in adata_real.obs.columns:
            print(f"  [SKIP] {ds_name}: no '{celltype_col}' column in obs")
            continue

        cell_types = adata_real.obs[celltype_col].unique()
        for ct in sorted(cell_types):
            fname = make_pred_vs_real_scatter(
                adata_real, adata_pred, ct, cyto, ds_name,
                pert_col, celltype_col, expr_dir,
            )
            if fname:
                print(f"    {ct}: saved pred-vs-real scatter")

        del adata_real, adata_pred

    # -----------------------------------------------------------------------
    # STEP 5: UMAPs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Generating UMAPs")
    print("=" * 60)

    # Atlas UMAP
    print("  Loading atlas for UMAP (subsampled) ...")
    atlas = ad.read_h5ad(ATLAS_PATH)
    make_umap_for_dataset(atlas, "cytokine_atlas", ATLAS_CELLTYPE_COL, ATLAS_PERT_COL,
                          umap_dir, max_cells=30000)
    del atlas
    print("  Atlas UMAP done.")

    # External dataset UMAPs
    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        # UMAP from the clustered (un-aligned) version for clean biology
        clustered_path = ds_info["clustered_h5ad"]
        if clustered_path.exists():
            print(f"  {ds_name}: loading clustered h5ad for UMAP ...")
            adata_clust = ad.read_h5ad(clustered_path)
            pert_col = ds_info["pert_col"]
            ct_col = "cell_type"

            # Check column availability
            if ct_col not in adata_clust.obs.columns:
                for candidate in ["cell_type_manual", "leiden", "louvain"]:
                    if candidate in adata_clust.obs.columns:
                        ct_col = candidate
                        break

            if pert_col not in adata_clust.obs.columns:
                for candidate in ["cytokine", "condition", "Condition", "StimType", "stim"]:
                    if candidate in adata_clust.obs.columns:
                        pert_col = candidate
                        break

            make_umap_for_dataset(adata_clust, f"{ds_name}_real", ct_col, pert_col, umap_dir)
            del adata_clust

        # Joint real+pred UMAP from aligned data
        real_path = ds_info["real_h5ad"]
        pred_path = ds_info["pred_h5ad"]
        if real_path.exists() and pred_path.exists():
            print(f"  {ds_name}: joint real+pred UMAP ...")
            adata_real = ad.read_h5ad(real_path)
            adata_pred = ad.read_h5ad(pred_path)

            if "is_imputed" in adata_real.var.columns:
                keep = ~adata_real.var["is_imputed"].astype(bool).values
                adata_real = adata_real[:, keep].copy()
                adata_pred = adata_pred[:, keep].copy()

            adata_real.obs["source"] = "real"
            adata_pred.obs["source"] = "predicted"

            # Ensure matching obs columns
            for col in ["cell_type", ds_info["pert_col"]]:
                if col in adata_real.obs.columns and col not in adata_pred.obs.columns:
                    adata_pred.obs[col] = adata_real.obs[col].values

            combined = ad.concat([adata_real, adata_pred], join="inner")
            pert_col = ds_info["pert_col"]
            ct_col = "cell_type"

            make_umap_for_dataset(combined, f"{ds_name}_real_vs_pred", ct_col, pert_col,
                                  umap_dir, source_col="source")
            del adata_real, adata_pred, combined

        print(f"  {ds_name}: UMAPs done.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE: All biological coherence analyses complete.")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTDIR}")
    print(f"  atlas_de_results.csv")
    print(f"  gene_coherence_metrics.csv")
    print(f"  plots/volcanos/")
    print(f"  plots/logfc_scatters/")
    print(f"  plots/pred_vs_real_expression/")
    print(f"  plots/umaps/")
    print(f"  plots/heatmaps/")


if __name__ == "__main__":
    main()
