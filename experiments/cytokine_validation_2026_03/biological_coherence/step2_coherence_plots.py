#!/usr/bin/env python3
"""
Step 2: Compute gene-level coherence metrics between atlas DE and external
predicted DE (non-imputed genes), and generate volcano plots, logFC scatter
plots, and heatmap summaries.

Requires: atlas_de_results.csv from step 1.

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step2_coherence_plots.py
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
RUN = BASE / "run_2026_02_11_all"
OUTDIR = BASE / "biological_coherence"
ATLAS_DE_PATH = OUTDIR / "atlas_de_results.csv"
FDR_THRESH = 0.05

EXTERNAL_DATASETS = {
    "GSE96583": {
        "eval_no_imputed": RUN / "GSE96583_eval/results_eval_GSE96583_batch2_IFNbeta_aligned_avg_predictions_no_imputed_genes",
    },
    "GSE178429": {
        "eval_no_imputed": RUN / "GSE178429_eval/results_eval_GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_predictions_no_imputed_genes",
    },
}


def _safe_log2fc(fc: pd.Series) -> pd.Series:
    fc = fc.copy()
    fc[fc <= 0] = np.nan
    return np.log2(fc)


def load_de_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["log2FC"] = _safe_log2fc(df["fold_change"])
    return df


def collect_cell_types_from_eval(eval_dir: Path) -> list[str]:
    cts = []
    for f in eval_dir.glob("*_real_de.csv"):
        ct = f.name.replace("_real_de.csv", "")
        cts.append(ct)
    return sorted(cts)


# ---------------------------------------------------------------------------
# Coherence metrics
# ---------------------------------------------------------------------------

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


def _find_atlas_cytokine(cyto: str, atlas_cytokines: np.ndarray) -> str | None:
    """Match external perturbation name to atlas cytokine name."""
    if cyto in atlas_cytokines:
        return cyto
    key = cyto.lower().replace("-", "").replace(" ", "")
    for c in atlas_cytokines:
        if key in c.lower().replace("-", "").replace(" ", ""):
            return c
    return None


def compute_gene_coherence(
    atlas_de: pd.DataFrame,
    ext_eval_dir: Path,
    dataset_name: str,
    variant: str = "avg_no_imputed",
) -> list[CoherenceResult]:
    results = []
    cell_types = collect_cell_types_from_eval(ext_eval_dir)
    atlas_cytokines = atlas_de["cytokine"].unique()

    for ct in cell_types:
        pred_de_path = ext_eval_dir / f"{ct}_pred_de.csv"
        real_de_path = ext_eval_dir / f"{ct}_real_de.csv"
        if not pred_de_path.exists():
            continue

        pred_de = load_de_table(pred_de_path)
        real_de = load_de_table(real_de_path) if real_de_path.exists() else None

        for cyto in pred_de["target"].unique():
            atlas_cyto = _find_atlas_cytokine(cyto, atlas_cytokines)
            if atlas_cyto is None:
                continue

            atlas_ct = atlas_de[(atlas_de["cytokine"] == atlas_cyto) &
                                (atlas_de["cell_type"] == ct)]
            if atlas_ct.empty:
                continue

            pred_cyto = pred_de[pred_de["target"] == cyto]
            merged = pd.merge(
                atlas_ct[["gene", "logFC_atlas", "fdr_atlas"]],
                pred_cyto[["feature", "log2FC", "fdr"]].rename(
                    columns={"feature": "gene", "log2FC": "logFC_pred", "fdr": "fdr_pred"}),
                on="gene", how="inner"
            ).dropna(subset=["logFC_atlas", "logFC_pred"])

            if len(merged) < 10:
                continue

            r = CoherenceResult(
                dataset=dataset_name, cell_type=ct, cytokine=cyto, variant=variant,
                n_shared_genes=len(merged),
            )

            try:
                r.spearman_logFC = spearmanr(merged["logFC_atlas"], merged["logFC_pred"])[0]
                r.pearson_logFC = pearsonr(merged["logFC_atlas"], merged["logFC_pred"])[0]
            except Exception:
                pass

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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_volcano_plot(de_df, atlas_de_genes, cell_type, cytokine, dataset_name, outdir):
    from adjustText import adjust_text

    df = de_df.copy()
    df["neg_log10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))
    df["log2FC"] = _safe_log2fc(df["fold_change"])
    df = df.dropna(subset=["log2FC", "neg_log10_fdr"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sig = df["fdr"] < FDR_THRESH

    ax.scatter(df.loc[~sig, "log2FC"], df.loc[~sig, "neg_log10_fdr"],
               s=4, alpha=0.3, c="grey", label="Not sig")
    ax.scatter(df.loc[sig, "log2FC"], df.loc[sig, "neg_log10_fdr"],
               s=6, alpha=0.5, c="steelblue", label=f"Pred DE (FDR<{FDR_THRESH})")

    texts = []
    if atlas_de_genes:
        in_atlas = df[sig & df["feature"].isin(atlas_de_genes)]
        ax.scatter(in_atlas["log2FC"], in_atlas["neg_log10_fdr"], s=12, alpha=0.8,
                   c="firebrick", label=f"Also DE in atlas ({len(in_atlas)})", zorder=5)
        top = in_atlas.nlargest(10, "neg_log10_fdr")
        for _, row in top.iterrows():
            texts.append(ax.text(row["log2FC"], row["neg_log10_fdr"],
                                 row["feature"], fontsize=6, ha="center", va="bottom"))

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.axhline(-np.log10(FDR_THRESH), ls="--", c="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("log2(fold change)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"{dataset_name}  {cell_type} — {cytokine}")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()

    ct_safe = cell_type.replace(" ", "_").replace("/", "-")
    fname = outdir / f"volcano_{dataset_name}_{ct_safe}_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def make_logfc_scatter(merged, cell_type, cytokine, dataset_name, outdir):
    df = merged.dropna(subset=["logFC_atlas", "logFC_pred"]).copy()
    if len(df) < 5:
        return None

    atlas_sig = df["fdr_atlas"] < FDR_THRESH
    pred_sig = df["fdr_pred"] < FDR_THRESH

    df["category"] = "Non-DE"
    df.loc[atlas_sig & ~pred_sig, "category"] = "Atlas only"
    df.loc[~atlas_sig & pred_sig, "category"] = "Pred only"
    df.loc[atlas_sig & pred_sig, "category"] = "Both"

    colors = {"Non-DE": "lightgrey", "Atlas only": "royalblue",
              "Pred only": "orange", "Both": "firebrick"}

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
    ax.set_title(f"{dataset_name} {cell_type} — {cytokine}\nSpearman={r_sp:.2f}, Pearson={r_pe:.2f}")
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()

    ct_safe = cell_type.replace(" ", "_").replace("/", "-")
    fname = outdir / f"scatter_logFC_{dataset_name}_{ct_safe}_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def make_logfc_scatter_atlas_vs_real(
    atlas_ct: pd.DataFrame,
    real_cyto: pd.DataFrame,
    cell_type: str,
    cytokine: str,
    dataset_name: str,
    outdir: Path,
) -> Path | None:
    """Scatter: atlas logFC (x) vs real external dataset logFC (y)."""
    real_renamed = real_cyto[["feature", "log2FC", "fdr"]].rename(
        columns={"feature": "gene", "log2FC": "logFC_real", "fdr": "fdr_real"}
    )
    merged = pd.merge(
        atlas_ct[["gene", "logFC_atlas", "fdr_atlas"]],
        real_renamed,
        on="gene",
        how="inner",
    ).dropna(subset=["logFC_atlas", "logFC_real"])
    if len(merged) < 5:
        return None

    atlas_sig = merged["fdr_atlas"] < FDR_THRESH
    real_sig = merged["fdr_real"] < FDR_THRESH

    merged = merged.copy()
    merged["category"] = "Non-DE"
    merged.loc[atlas_sig & ~real_sig, "category"] = "Atlas only"
    merged.loc[~atlas_sig & real_sig, "category"] = "Real only"
    merged.loc[atlas_sig & real_sig, "category"] = "Both"

    colors = {"Non-DE": "lightgrey", "Atlas only": "royalblue",
              "Real only": "forestgreen", "Both": "firebrick"}

    r_sp = spearmanr(merged["logFC_atlas"], merged["logFC_real"])[0]
    r_pe = pearsonr(merged["logFC_atlas"], merged["logFC_real"])[0]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for cat in ["Non-DE", "Atlas only", "Real only", "Both"]:
        sub = merged[merged["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub["logFC_atlas"], sub["logFC_real"], s=6, alpha=0.5,
                   c=colors[cat], label=f"{cat} ({len(sub)})")

    lims = [min(merged["logFC_atlas"].min(), merged["logFC_real"].min()),
            max(merged["logFC_atlas"].max(), merged["logFC_real"].max())]
    ax.plot(lims, lims, "--", c="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("logFC (atlas)")
    ax.set_ylabel("logFC (real external)")
    ax.set_title(f"{dataset_name} {cell_type} — {cytokine}\nAtlas vs real external dataset\nSpearman={r_sp:.2f}, Pearson={r_pe:.2f}")
    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()

    ct_safe = cell_type.replace(" ", "_").replace("/", "-")
    fname = outdir / f"scatter_logFC_atlas_vs_real_external_dataset_{dataset_name}_{ct_safe}_{cytokine.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def make_coherence_heatmap(coherence_df, metric, outdir, dataset_name=None):
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
    vmin = -1 if "spearman" in metric or "pearson" in metric else 0
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=vmin, vmax=1,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    title = f"{dataset_name}: {metric}" if dataset_name else metric
    ax.set_title(title)
    plt.tight_layout()

    ds_tag = dataset_name or "all"
    fname = outdir / f"heatmap_{metric}_{ds_tag}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plots_dir = OUTDIR / "plots"
    volcano_dir = plots_dir / "volcanos"; volcano_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir = plots_dir / "logfc_scatters"; scatter_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = plots_dir / "heatmaps"; heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Load atlas DE
    if not ATLAS_DE_PATH.exists():
        print(f"ERROR: Atlas DE not found at {ATLAS_DE_PATH}. Run step1 first.")
        return
    atlas_de = pd.read_csv(ATLAS_DE_PATH)
    print(f"Loaded atlas DE: {len(atlas_de)} rows, "
          f"{atlas_de['cytokine'].nunique()} cytokines, "
          f"{atlas_de['cell_type'].nunique()} cell types")

    # --- Coherence metrics ---
    print("\n--- Computing gene-level coherence metrics ---")
    all_coherence = []
    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        eval_dir = ds_info["eval_no_imputed"]
        if not eval_dir.exists():
            print(f"  [SKIP] {ds_name}: {eval_dir} not found")
            continue
        results = compute_gene_coherence(atlas_de, eval_dir, ds_name)
        all_coherence.extend(results)
        print(f"  {ds_name}: {len(results)} coherence results")

    if all_coherence:
        coherence_df = pd.DataFrame([vars(r) for r in all_coherence])
        coherence_path = OUTDIR / "gene_coherence_metrics.csv"
        coherence_df.to_csv(coherence_path, index=False)
        print(f"\nSaved to {coherence_path}")
        print(coherence_df[["dataset", "cell_type", "cytokine", "spearman_logFC",
                            "jaccard_de", "precision_de", "recall_de"]].to_string(index=False))
    else:
        coherence_df = pd.DataFrame()
        print("WARNING: No coherence results.")

    # --- Volcano + logFC scatter plots ---
    print("\n--- Generating volcano & logFC scatter plots ---")
    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        eval_dir = ds_info["eval_no_imputed"]
        if not eval_dir.exists():
            continue
        for ct in collect_cell_types_from_eval(eval_dir):
            pred_de_path = eval_dir / f"{ct}_pred_de.csv"
            real_de_path = eval_dir / f"{ct}_real_de.csv"
            if not pred_de_path.exists():
                continue
            pred_de = load_de_table(pred_de_path)
            real_de = load_de_table(real_de_path) if real_de_path.exists() else None

            for cyto in pred_de["target"].unique():
                pred_cyto = pred_de[pred_de["target"] == cyto]
                atlas_cyto = _find_atlas_cytokine(cyto, atlas_de["cytokine"].unique())

                atlas_sig_genes = None
                if atlas_cyto:
                    atlas_ct = atlas_de[(atlas_de["cytokine"] == atlas_cyto) &
                                        (atlas_de["cell_type"] == ct)]
                    if not atlas_ct.empty:
                        atlas_sig_genes = set(atlas_ct.loc[atlas_ct["fdr_atlas"] < FDR_THRESH, "gene"])

                make_volcano_plot(pred_cyto, atlas_sig_genes, ct, cyto, ds_name, volcano_dir)

                if atlas_cyto:
                    atlas_ct = atlas_de[(atlas_de["cytokine"] == atlas_cyto) &
                                        (atlas_de["cell_type"] == ct)]
                    if not atlas_ct.empty:
                        merged = pd.merge(
                            atlas_ct[["gene", "logFC_atlas", "fdr_atlas"]],
                            pred_cyto[["feature", "log2FC", "fdr"]].rename(
                                columns={"feature": "gene", "log2FC": "logFC_pred", "fdr": "fdr_pred"}),
                            on="gene", how="inner"
                        )
                        make_logfc_scatter(merged, ct, cyto, ds_name, scatter_dir)

                        if real_de is not None:
                            real_cyto = real_de[real_de["target"] == cyto]
                            if not real_cyto.empty:
                                make_logfc_scatter_atlas_vs_real(
                                    atlas_ct, real_cyto, ct, cyto, ds_name, scatter_dir
                                )

                print(f"  {ds_name}/{ct}/{cyto}: done")

    # --- Heatmaps ---
    if not coherence_df.empty:
        print("\n--- Generating heatmaps ---")
        for ds_name in coherence_df["dataset"].unique():
            for metric in ["spearman_logFC", "jaccard_de", "precision_de", "recall_de"]:
                make_coherence_heatmap(coherence_df, metric, heatmap_dir, ds_name)
        print("  Heatmaps done.")

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
