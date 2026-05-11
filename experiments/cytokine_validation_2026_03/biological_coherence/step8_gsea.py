#!/usr/bin/env python3
"""
Step 8: GSEA pathway enrichment using gseapy prerank.

Three levels of comparison:
  Level 1: Atlas real DE vs CV predicted DE (in-distribution)
  Level 2: Atlas real DE vs external predicted DE (cross-dataset)
  Level 3: External real DE vs external predicted DE

Gene sets: MSigDB Hallmark + GO Biological Process

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step8_gsea.py --level 2
    uv run python experiments/cytokine_validation_2026/biological_coherence/step8_gsea.py --level 3
    # Level 1 needs cv_pred_de_results.csv (from step7b, SLURM job)
    uv run python experiments/cytokine_validation_2026/biological_coherence/step8_gsea.py --level 1
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
BIO_DIR = BASE / "biological_coherence"
OUTDIR = BIO_DIR / "plots" / "gsea"
RUN = BASE / "run_2026_02_11_all"

ATLAS_DE_PATH = BIO_DIR / "atlas_de_results.csv"
CV_PRED_DE_PATH = BIO_DIR / "cv_pred_de_results.csv"

GENE_SETS = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2021"]

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


def run_prerank(rnk: pd.Series, label: str, outdir: Path) -> pd.DataFrame | None:
    """Run gseapy prerank and return results DataFrame."""
    rnk = rnk.dropna().sort_values(ascending=False)
    rnk = rnk[~rnk.index.duplicated(keep="first")]
    if len(rnk) < 50:
        print(f"    Skipping {label}: only {len(rnk)} genes")
        return None

    try:
        result = gp.prerank(
            rnk=rnk,
            gene_sets=GENE_SETS,
            outdir=str(outdir / label),
            min_size=10,
            max_size=500,
            permutation_num=1000,
            seed=42,
            verbose=False,
            no_plot=True,
        )
        res = result.res2d
        res["source"] = label
        return res
    except Exception as e:
        print(f"    GSEA failed for {label}: {e}")
        return None


def compute_de_inline(adata, pert_col, control_label, cell_type):
    """Compute DE for a single cell type inline (for external datasets)."""
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

    df = sc.get.rank_genes_groups_df(sub, group="perturbed")
    return df.set_index("names")["logfoldchanges"]


def _clean_term(term: str) -> str:
    """Strip gene-set database prefixes, clean underscores, keep GO IDs."""
    term = re.sub(r"^GO_Biological_Process_\d{4}__", "", term)
    term = re.sub(r"^MSigDB_Hallmark_\d{4}__", "Hallmark: ", term)
    term = term.replace("_", " ")
    term = term.replace("Normalized", "Normalised")
    term = term.replace("normalized", "normalised")
    if term and term[0].islower():
        term = term[0].upper() + term[1:]
    return term.strip()


def make_comparison_dotplot(results_real, results_pred, label_real, label_pred, title, outpath, top_n=20):
    """Dot plot anchored on real-data significant pathways.

    Only pathways significant (FDR < 0.25) in the real data are shown.
    If the real data has no significant pathways the plot is skipped
    entirely — there is no confirmed treatment effect to evaluate.
    """
    if results_real is None or results_pred is None:
        return
    if results_real.empty or results_pred.empty:
        return

    sig_real = results_real[results_real["FDR q-val"] < 0.25].sort_values(
        "NES", key=abs, ascending=False
    ).head(top_n)

    if sig_real.empty:
        print(f"    Skipping plot — no significant pathways in real data")
        return

    raw_terms = list(sig_real["Term"])
    display_terms = [_clean_term(t) for t in raw_terms]

    rows = []
    for t in raw_terms:
        for src_label, src_df in [(label_real, results_real), (label_pred, results_pred)]:
            match = src_df[src_df["Term"] == t]
            if not match.empty:
                row = match.iloc[0]
                rows.append({
                    "Term": t, "Source": src_label,
                    "NES": float(row["NES"]),
                })
            else:
                rows.append({"Term": t, "Source": src_label, "NES": 0.0})

    plot_df = pd.DataFrame(rows)
    col_real, col_pred = "#2166ac", "#b2182b"

    n_terms = len(raw_terms)
    # Smaller figure when few pathways to reduce white space
    if n_terms <= 5:
        w, h = 7, max(1.8, n_terms * 0.5)
    else:
        w, h = 10, max(4, n_terms * 0.4)
    fig, ax = plt.subplots(figsize=(w, h))
    offset = 0.15

    for src_label, color, dy in [(label_real, col_real, -offset), (label_pred, col_pred, offset)]:
        sub = plot_df[plot_df["Source"] == src_label]
        y_pos = np.array([raw_terms.index(t) for t in sub["Term"]]) + dy
        sizes = np.clip(np.abs(sub["NES"].values) * 60, 30, 350)
        ax.scatter(
            sub["NES"].values, y_pos, s=sizes, c=color, edgecolors="white",
            linewidths=0.5, alpha=0.85, label=src_label, zorder=3,
        )

    ax.set_yticks(range(len(display_terms)))
    ax.set_yticklabels(display_terms, fontsize=7)
    ax.set_xlabel("Normalised Enrichment Score (NES)")
    ax.axvline(0, color="gray", lw=0.5)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout(pad=0.8, h_pad=0.5)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def level1_atlas_vs_cv(outdir):
    """Atlas real DE vs CV predicted DE."""
    print("\n=== Level 1: Atlas real vs CV predicted DE ===")
    if not CV_PRED_DE_PATH.exists():
        print(f"  CV pred DE not found at {CV_PRED_DE_PATH}. Run step7b first.")
        return

    atlas_de = pd.read_csv(ATLAS_DE_PATH)
    cv_de = pd.read_csv(CV_PRED_DE_PATH)

    focus_cytokines = ["IFN-beta", "IFN-gamma", "TNF-alpha"]
    cell_types = sorted(set(atlas_de["cell_type"]) & set(cv_de["cell_type"]))

    all_results = []
    for cyto in focus_cytokines:
        atlas_sub = atlas_de[atlas_de["cytokine"] == cyto]
        cv_sub = cv_de[cv_de["cytokine"] == cyto]
        if atlas_sub.empty or cv_sub.empty:
            continue

        for ct in cell_types:
            a = atlas_sub[atlas_sub["cell_type"] == ct]
            c = cv_sub[cv_sub["cell_type"] == ct]
            if a.empty or c.empty:
                continue

            rnk_atlas = a.set_index("gene")["logFC_atlas"].dropna()
            rnk_cv = c.set_index("gene")["logFC_cv_pred"].dropna()

            safe = f"{ct}_{cyto}".replace(" ", "_")
            res_a = run_prerank(rnk_atlas, f"atlas_{safe}", outdir)
            res_c = run_prerank(rnk_cv, f"cv_pred_{safe}", outdir)

            if res_a is not None:
                all_results.append(res_a)
            if res_c is not None:
                all_results.append(res_c)

            make_comparison_dotplot(
                res_a, res_c, "Atlas real", "CV predicted",
                f"GSEA: {ct} — {cyto}\nAtlas real vs CV predicted",
                outdir / f"gsea_compare_l1_{safe}.png",
            )

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(outdir / "gsea_level1_results.csv", index=False)
        print(f"  Saved gsea_level1_results.csv ({len(combined)} rows)")


def level2_atlas_vs_external_pred(outdir):
    """Atlas real DE vs external predicted DE."""
    print("\n=== Level 2: Atlas real vs external predicted DE ===")
    atlas_de = pd.read_csv(ATLAS_DE_PATH)

    all_results = []
    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        print(f"\n  Dataset: {ds_name}")
        cyto = ds_info["atlas_cytokine"]
        atlas_sub = atlas_de[(atlas_de["cytokine"] == cyto)]

        adata_real = ad.read_h5ad(ds_info["real_h5ad"])
        adata_pred = ad.read_h5ad(ds_info["pred_h5ad"])

        if "is_imputed" in adata_real.var.columns:
            keep = ~adata_real.var["is_imputed"].astype(bool).values
            adata_pred = adata_pred[:, keep].copy()

        pert_col = ds_info["pert_col"]
        ctrl = ds_info["control_label"]
        ct_col = "cell_type"
        if ct_col not in adata_pred.obs.columns:
            adata_pred.obs[ct_col] = adata_real.obs[ct_col].values
        if pert_col not in adata_pred.obs.columns:
            adata_pred.obs[pert_col] = adata_real.obs[pert_col].values

        cell_types = sorted(adata_pred.obs[ct_col].unique())

        for ct in cell_types:
            a = atlas_sub[atlas_sub["cell_type"] == ct]
            if a.empty:
                continue

            rnk_atlas = a.set_index("gene")["logFC_atlas"].dropna()
            rnk_pred = compute_de_inline(adata_pred, pert_col, ctrl, ct)
            if rnk_pred is None:
                continue

            safe = f"{ds_name}_{ct}_{cyto}".replace(" ", "_")
            res_a = run_prerank(rnk_atlas, f"atlas_{safe}", outdir)
            res_p = run_prerank(rnk_pred, f"ext_pred_{safe}", outdir)

            if res_a is not None:
                all_results.append(res_a)
            if res_p is not None:
                all_results.append(res_p)

            make_comparison_dotplot(
                res_a, res_p, "Atlas real", f"{ds_name} predicted",
                f"GSEA: {ct} — {cyto}\nAtlas real vs {ds_name} predicted",
                outdir / f"gsea_compare_l2_{safe}.png",
            )

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(outdir / "gsea_level2_results.csv", index=False)
        print(f"  Saved gsea_level2_results.csv ({len(combined)} rows)")


def level3_external_real_vs_pred(outdir):
    """External real DE vs external predicted DE."""
    print("\n=== Level 3: External real vs external predicted DE ===")

    all_results = []
    for ds_name, ds_info in EXTERNAL_DATASETS.items():
        print(f"\n  Dataset: {ds_name}")
        adata_real = ad.read_h5ad(ds_info["real_h5ad"])
        adata_pred = ad.read_h5ad(ds_info["pred_h5ad"])

        if "is_imputed" in adata_real.var.columns:
            keep = ~adata_real.var["is_imputed"].astype(bool).values
            adata_real = adata_real[:, keep].copy()
            adata_pred = adata_pred[:, keep].copy()

        pert_col = ds_info["pert_col"]
        ctrl = ds_info["control_label"]
        ct_col = "cell_type"
        if ct_col not in adata_pred.obs.columns:
            adata_pred.obs[ct_col] = adata_real.obs[ct_col].values
        if pert_col not in adata_pred.obs.columns:
            adata_pred.obs[pert_col] = adata_real.obs[pert_col].values

        # Normalize real data for DE
        X_sample = _to_dense(adata_real.X[:100])
        if (X_sample >= 0).all() and np.allclose(X_sample, X_sample.astype(int)):
            _normalize_sparse_inplace(adata_real)

        cell_types = sorted(adata_real.obs[ct_col].unique())
        cyto = ds_info["atlas_cytokine"]

        for ct in cell_types:
            rnk_real = compute_de_inline(adata_real, pert_col, ctrl, ct)
            rnk_pred = compute_de_inline(adata_pred, pert_col, ctrl, ct)
            if rnk_real is None or rnk_pred is None:
                continue

            safe = f"{ds_name}_{ct}_{cyto}".replace(" ", "_")
            res_r = run_prerank(rnk_real, f"ext_real_{safe}", outdir)
            res_p = run_prerank(rnk_pred, f"ext_pred_{safe}", outdir)

            if res_r is not None:
                all_results.append(res_r)
            if res_p is not None:
                all_results.append(res_p)

            make_comparison_dotplot(
                res_r, res_p, f"{ds_name} real", f"{ds_name} predicted",
                f"GSEA: {ct} — {cyto}\n{ds_name} real vs predicted",
                outdir / f"gsea_compare_l3_{safe}.png",
            )

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(outdir / "gsea_level3_results.csv", index=False)
        print(f"  Saved gsea_level3_results.csv ({len(combined)} rows)")


def _parse_source(source: str):
    """Extract (side, key) from source string like 'ext_real_GSE96583_B_Naive_IFN-beta'."""
    for prefix, side in [
        ("ext_real_", "real"), ("ext_pred_", "pred"),
        ("atlas_", "real"), ("cv_pred_", "pred"),
    ]:
        if source.startswith(prefix):
            return side, source[len(prefix):]
    return None, source


def _split_key(key: str):
    """Extract dataset name and clean display label from a key like 'GSE96583_CD4_Memory_IFN-beta'."""
    m = re.match(r"(GSE\d+)_(.*)", key)
    if m:
        return m.group(1), m.group(2).replace("_", " ")
    return "", key.replace("_", " ")


def nes_scatter_from_csv(csv_path: Path, level_label: str, outdir: Path, *, comparison: str = ""):
    """Build real-vs-predicted NES scatter plots from a saved GSEA results CSV."""
    if not csv_path.exists():
        print(f"  {csv_path.name} not found, skipping")
        return

    df = pd.read_csv(csv_path)
    df[["side", "key"]] = df["source"].apply(lambda s: pd.Series(_parse_source(s)))
    df = df[df["side"].notna()]

    real_df = df[df["side"] == "real"][["Term", "NES", "key"]].rename(columns={"NES": "NES_real"})
    pred_df = df[df["side"] == "pred"][["Term", "NES", "key"]].rename(columns={"NES": "NES_pred"})
    merged = real_df.merge(pred_df, on=["Term", "key"], how="inner")

    if merged.empty:
        print(f"  No matched real/pred pairs in {csv_path.name}")
        return

    keys = sorted(merged["key"].unique())

    for key in keys:
        sub = merged[merged["key"] == key]
        if len(sub) < 5:
            continue

        x = sub["NES_real"].values
        y = sub["NES_pred"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        x, y = x[mask], y[mask]

        from scipy.stats import pearsonr, spearmanr
        r_p = pearsonr(x, y)[0]
        r_s = spearmanr(x, y)[0]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, s=12, alpha=0.6, c="steelblue", edgecolors="none", rasterized=True)

        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo, hi], [lo, hi], "--", c="red", lw=1, alpha=0.7)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel("Real NES")
        ax.set_ylabel("Predicted NES")
        safe_key = key.replace(" ", "_")
        ds, display_key = _split_key(key)
        comp_filled = comparison.replace("{ds}", ds) if comparison and ds else comparison
        comp_line = f" ({comp_filled})" if comp_filled else ""
        ax.set_title(f"NES{comp_line}\n{display_key}  r = {r_p:.3f}", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        fig.savefig(outdir / f"nes_scatter_{level_label}_{safe_key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved nes_scatter_{level_label}_{safe_key}.png  (r_p={r_p:.3f}, r_s={r_s:.3f})")

    real_fdr = df[df["side"] == "real"][["Term", "FDR q-val", "key"]].rename(
        columns={"FDR q-val": "FDR_real"}
    )
    merged_full = merged.merge(real_fdr, on=["Term", "key"], how="left")
    merged_full["FDR_real"] = merged_full["FDR_real"].fillna(1.0)

    fin = np.isfinite(merged_full["NES_real"]) & np.isfinite(merged_full["NES_pred"])
    mf = merged_full[fin].copy()

    if len(mf) < 5:
        return

    all_x = mf["NES_real"].values
    all_y = mf["NES_pred"].values
    from scipy.stats import pearsonr, spearmanr
    r_p = pearsonr(all_x, all_y)[0]
    r_s = spearmanr(all_x, all_y)[0]

    def _axis_setup(ax, x, y, xlabel="Real NES", ylabel="Predicted NES"):
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo, hi], [lo, hi], "--", c="grey", lw=1, alpha=0.5)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="grey", lw=0.3, alpha=0.5)
        ax.axvline(0, color="grey", lw=0.3, alpha=0.5)
        ax.set_aspect("equal", adjustable="datalim")

    subtitle = f"r = {r_p:.3f}"
    ds_names = sorted({_split_key(k)[0] for k in keys if _split_key(k)[0]})
    ds_str = " + ".join(ds_names) if ds_names else "external"
    comp_filled = comparison.replace("{ds}", ds_str) if comparison else ""
    comp_line = f" ({comp_filled})" if comp_filled else ""

    # --- Plain ALL scatter ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_x, all_y, s=8, alpha=0.4, c="steelblue", edgecolors="none", rasterized=True)
    _axis_setup(ax, all_x, all_y)
    ax.set_title(f"NES{comp_line} — all cell types\n{subtitle}", fontsize=10)
    plt.tight_layout()
    fig.savefig(outdir / f"nes_scatter_{level_label}_ALL.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nes_scatter_{level_label}_ALL.png  (r_p={r_p:.3f}, r_s={r_s:.3f})")

    # --- Version A: colour by real FDR ---
    fig, ax = plt.subplots(figsize=(7, 6))
    fdr = mf["FDR_real"].values
    fdr_clipped = np.clip(fdr, 1e-10, 1.0)
    from matplotlib.colors import LogNorm
    sc = ax.scatter(
        all_x, all_y, s=10, alpha=0.6,
        c=fdr_clipped, cmap="RdYlGn", norm=LogNorm(vmin=1e-4, vmax=1.0),
        edgecolors="none", rasterized=True,
    )
    _axis_setup(ax, all_x, all_y)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Real FDR q-val", fontsize=9)
    ax.set_title(f"NES{comp_line} — coloured by real FDR\n{subtitle}", fontsize=10)
    plt.tight_layout()
    fig.savefig(outdir / f"nes_scatter_{level_label}_ALL_byFDR.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nes_scatter_{level_label}_ALL_byFDR.png")

    # --- Version B: colour by direction agreement ---
    cats = np.where(
        (all_x > 0) & (all_y > 0), "correct up",
        np.where(
            (all_x < 0) & (all_y < 0), "correct down",
            "wrong direction",
        ),
    )
    cat_colors = {"correct up": "#2166ac", "correct down": "#4dac26", "wrong direction": "#d6604d"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for cat in ["wrong direction", "correct down", "correct up"]:
        m = cats == cat
        if m.any():
            ax.scatter(
                all_x[m], all_y[m], s=10, alpha=0.55,
                c=cat_colors[cat], edgecolors="none", label=cat, rasterized=True,
            )
    _axis_setup(ax, all_x, all_y)
    n_correct = ((cats == "correct up") | (cats == "correct down")).sum()
    pct = n_correct / len(cats) * 100
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_title(
        f"NES{comp_line} — direction agreement\n"
        f"{subtitle}, correct direction: {pct:.0f}%", fontsize=10
    )
    plt.tight_layout()
    fig.savefig(outdir / f"nes_scatter_{level_label}_ALL_byDirection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nes_scatter_{level_label}_ALL_byDirection.png  ({pct:.0f}% correct)")

    # --- Version C: colour by pathway family ---
    family_patterns = [
        ("Interferon response", re.compile(r"interferon|IFN|antiviral|viral", re.I)),
        ("Immune signalling", re.compile(r"immune|innate|adaptive|complement|MHC|antigen|lymphocyte|leukocyte", re.I)),
        ("Inflammatory / NF-kB", re.compile(r"inflam|NF.?kB|TNF|IL.?\d|cytokine|interleukin", re.I)),
        ("Chemotaxis / migration", re.compile(r"chemotax|chemokine|migrat|adhesion", re.I)),
        ("Cell cycle / proliferation", re.compile(r"cell.?cycle|mitot|prolifer|G2.?M|E2F|checkpoint|DNA.?repair", re.I)),
        ("Metabolism", re.compile(r"metabol|glycoly|oxidative|mTOR|fatty.?acid|cholesterol|adipog", re.I)),
        ("Apoptosis / stress", re.compile(r"apoptos|programmed.?cell.?death|p53|hypoxia|reactive.?oxygen|stress", re.I)),
        ("Signalling", re.compile(r"signal|kinase|phosphory|MAPK|JAK|STAT|Wnt|Notch|Hedgehog|PI3K|RAS", re.I)),
    ]

    def _classify_term(term: str) -> str:
        for name, pat in family_patterns:
            if pat.search(term):
                return name
        return "Other"

    families = np.array([_classify_term(t) for t in mf["Term"].values])
    unique_fams = sorted(set(families))
    palette_c = plt.cm.Set2(np.linspace(0, 1, max(len(unique_fams), 3)))
    fam_cmap = {f: palette_c[i] for i, f in enumerate(unique_fams)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for fam in unique_fams:
        m = families == fam
        if m.any():
            ax.scatter(
                all_x[m], all_y[m], s=10, alpha=0.55,
                c=[fam_cmap[fam]], edgecolors="none", label=fam, rasterized=True,
            )
    _axis_setup(ax, all_x, all_y)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9, ncol=1)
    ax.set_title(f"NES{comp_line} — by pathway family\n{subtitle}", fontsize=10)
    plt.tight_layout()
    fig.savefig(outdir / f"nes_scatter_{level_label}_ALL_byFamily.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved nes_scatter_{level_label}_ALL_byFamily.png")


def _extract_dataset_celltype(key: str) -> tuple[str, str]:
    """From key like 'GSE96583_B_Naive_IFN-beta' return (GSE96583, B_Naive)."""
    key = key.replace("_IFN-beta", "").replace("_IFN-gamma", "")
    m = re.match(r"(GSE\d+)_(.*)", key)
    return (m.group(1), m.group(2).replace("_", " ")) if m else ("", key)


def make_joint_dotplot_agreed(
    df: pd.DataFrame,
    dataset: str,
    level: str,
    outpath: Path,
    *,
    label_real: str,
    label_pred: str,
    top_n_pathways: int = 25,
) -> None:
    """Dotplot: top agreed pathways (real & pred same direction + sig) x cell types.
    Two panels side-by-side: real | pred."""
    real_df = df[(df["dataset"] == dataset) & (df["side"] == "real")][["Term", "cell_type", "NES", "FDR q-val"]]
    pred_df = df[(df["dataset"] == dataset) & (df["side"] == "pred")][["Term", "cell_type", "NES", "FDR q-val"]]

    if real_df.empty or pred_df.empty:
        return

    merged = real_df.merge(
        pred_df,
        on=["Term", "cell_type"],
        how="inner",
        suffixes=("_real", "_pred"),
    )
    if merged.empty:
        return

    # Agreed = same direction (NES_real * NES_pred > 0) and both FDR<0.25
    concordant = (merged["NES_real"] * merged["NES_pred"]) > 0
    both_sig = (merged["FDR q-val_real"] < 0.25) & (merged["FDR q-val_pred"] < 0.25)
    agreed = merged[concordant & both_sig]

    if agreed.empty:
        # Fallback: concordant only, rank by product
        agreed = merged[concordant].copy()
        agreed["score"] = agreed["NES_real"] * agreed["NES_pred"]
    else:
        agreed = agreed.copy()
        agreed["score"] = agreed["NES_real"] * agreed["NES_pred"]

    # Pathway-level score: sum of (NES_real * NES_pred) over cell types
    pathway_scores = agreed.groupby("Term")["score"].sum().abs().sort_values(ascending=False)
    terms = pathway_scores.head(top_n_pathways).index.tolist()
    if not terms:
        return

    # Build pivots for real and pred
    real_piv = real_df.pivot_table(index="Term", columns="cell_type", values="NES", aggfunc="first")
    real_fdr = real_df.pivot_table(index="Term", columns="cell_type", values="FDR q-val", aggfunc="first")
    pred_piv = pred_df.pivot_table(index="Term", columns="cell_type", values="NES", aggfunc="first")
    pred_fdr = pred_df.pivot_table(index="Term", columns="cell_type", values="FDR q-val", aggfunc="first")

    cell_types = sorted(set(real_piv.columns) | set(pred_piv.columns))
    terms = [t for t in terms if t in real_piv.index or t in pred_piv.index]
    if not terms:
        return

    real_nes = real_piv.reindex(index=terms, columns=cell_types)
    real_fdr = real_fdr.reindex(index=terms, columns=cell_types)
    pred_nes = pred_piv.reindex(index=terms, columns=cell_types)
    pred_fdr = pred_fdr.reindex(index=terms, columns=cell_types)

    n_rows, n_cols = len(terms), len(cell_types)
    # Smaller figure when few pathways
    h = max(2.5, n_rows * 0.4) if n_rows <= 8 else max(5, n_rows * 0.35)
    w = max(8, n_cols * 1.0) if n_rows <= 5 else max(10, n_cols * 1.2)
    fig, (ax_real, ax_pred) = plt.subplots(1, 2, figsize=(w, h), sharey=True)

    def _draw_panel(ax, pivot_nes, pivot_fdr, title):
        neglog_fdr = -np.log10(np.clip(pivot_fdr.values, 1e-10, 1))
        neglog_fdr = np.clip(np.where(np.isfinite(neglog_fdr), neglog_fdr, 0), 0, 10)
        sizes = 30 + neglog_fdr * 25

        for j, ct in enumerate(cell_types):
            for i, term in enumerate(terms):
                nes = pivot_nes.loc[term, ct]
                if pd.isna(nes):
                    continue
                s = sizes[i, j]
                color = "#2166ac" if nes > 0 else "#b2182b"
                ax.scatter(j, i, s=s, c=color, alpha=0.85, edgecolors="white", linewidths=0.3)

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(cell_types, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(len(terms) - 0.5, -0.5)
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_xlabel("Cell type")
        ax.set_title(title)

    _draw_panel(ax_real, real_nes, real_fdr, label_real)
    _draw_panel(ax_pred, pred_nes, pred_fdr, label_pred)

    ax_real.set_yticks(range(len(terms)))
    ax_real.set_yticklabels([_clean_term(t) for t in terms], fontsize=7)

    fig.suptitle(
        f"GSEA {level}: {dataset} — top agreed pathways (real & pred)\nNES: blue=up, red=down; size=-log10 FDR",
        fontsize=10, y=1.02,
    )
    plt.tight_layout(pad=0.8)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def generate_joint_dotplots(outdir: Path) -> None:
    """Joint GSEA dotplots per level per dataset: top agreed pathways x cell types (2 panels: real | pred)."""
    print("\n=== Joint GSEA Dotplots (pathways x cell types) ===")

    configs = [
        ("gsea_level2_results.csv", "L2", "Atlas", "Predicted"),
        ("gsea_level3_results.csv", "L3", "Real", "Predicted"),
    ]

    for csv_name, level, label_real, label_pred in configs:
        csv_path = outdir / csv_name
        if not csv_path.exists():
            print(f"  {csv_name} not found, skipping {level}")
            continue

        df = pd.read_csv(csv_path)
        df[["side", "key"]] = df["source"].apply(lambda s: pd.Series(_parse_source(s)))
        df = df[df["side"].notna()]

        df[["dataset", "cell_type"]] = df["key"].apply(
            lambda k: pd.Series(_extract_dataset_celltype(k))
        )
        df["cell_type"] = df["cell_type"].str.replace(" ", "_")

        for ds in sorted(df["dataset"].unique()):
            if not ds.startswith("GSE"):
                continue
            make_joint_dotplot_agreed(
                df, ds, level,
                outdir / f"gsea_joint_{level}_{ds}.png",
                label_real=label_real,
                label_pred=label_pred,
            )


def generate_stacked_comparison_dotplots(outdir: Path) -> None:
    """Stacked GSEA comparison dotplots: one per (level, dataset), cell types stacked with labels."""
    print("\n=== Stacked GSEA Comparison Dotplots (cell-type labels left) ===")

    configs = [
        ("gsea_level2_results.csv", "L2", "Atlas", "Predicted"),
        ("gsea_level3_results.csv", "L3", "Real", "Predicted"),
    ]

    for csv_name, level, label_real, label_pred in configs:
        csv_path = outdir / csv_name
        if not csv_path.exists():
            print(f"  {csv_name} not found, skipping {level}")
            continue

        df = pd.read_csv(csv_path)
        df[["side", "key"]] = df["source"].apply(lambda s: pd.Series(_parse_source(s)))
        df = df[df["side"].notna()]

        df[["dataset", "cell_type"]] = df["key"].apply(
            lambda k: pd.Series(_extract_dataset_celltype(k))
        )
        df["cell_type"] = df["cell_type"].str.replace(" ", "_")

        for ds in sorted(df["dataset"].unique()):
            if not ds.startswith("GSE"):
                continue
            make_stacked_comparison_dotplot(
                df, ds, level,
                outdir / f"gsea_compare_stacked_{level}_{ds}.png",
                label_real=label_real,
                label_pred=label_pred,
                top_n_pathways=10,
            )


def make_stacked_comparison_dotplot(
    df: pd.DataFrame,
    dataset: str,
    level: str,
    outpath: Path,
    *,
    label_real: str,
    label_pred: str,
    top_n_pathways: int = 10,
) -> None:
    """Stacked comparison dotplots: one row per cell type, top agreed ENRICHED pathways (positive NES).
    Top N pathways overall (across all cell types), not per cell type.
    Same style as gsea_compare (pathways on y, NES on x, blue=real, red=pred)."""
    ds_df = df[(df["dataset"] == dataset)]
    cell_types = sorted(ds_df["cell_type"].unique())

    # Collect agreed ENRICHED (positive NES) pathways, track best FDR across all cell types
    # best_fdr[term] = min FDR (real or pred) across any cell type
    best_fdr = {}
    for ct in cell_types:
        real_ct = ds_df[(ds_df["cell_type"] == ct) & (ds_df["side"] == "real")][["Term", "NES", "FDR q-val"]]
        pred_ct = ds_df[(ds_df["cell_type"] == ct) & (ds_df["side"] == "pred")][["Term", "NES", "FDR q-val"]]
        if real_ct.empty or pred_ct.empty:
            continue
        merged = real_ct.merge(
            pred_ct, on="Term", how="inner", suffixes=("_real", "_pred")
        )
        enriched = (merged["NES_real"] > 0) & (merged["NES_pred"] > 0)
        agreed = merged[enriched]
        if agreed.empty:
            continue
        for _, row in agreed.iterrows():
            term = row["Term"]
            fdr = min(row["FDR q-val_real"], row["FDR q-val_pred"])
            if term not in best_fdr or fdr < best_fdr[term]:
                best_fdr[term] = fdr

    if not best_fdr:
        return

    # Top N pathways by best FDR (most significant)
    ranked = sorted(best_fdr.items(), key=lambda x: x[1])[:top_n_pathways]
    global_terms = [t for t, _ in ranked]

    # Build panels: (cell_type, real_df, pred_df) — same terms for all
    panels = []
    for ct in cell_types:
        real_ct = ds_df[(ds_df["cell_type"] == ct) & (ds_df["side"] == "real")][["Term", "NES", "FDR q-val"]]
        pred_ct = ds_df[(ds_df["cell_type"] == ct) & (ds_df["side"] == "pred")][["Term", "NES", "FDR q-val"]]
        if real_ct.empty or pred_ct.empty:
            continue
        panels.append((ct, real_ct, pred_ct, global_terms))

    if not panels:
        return

    n_rows = len(panels)
    n_pathways = len(panels[0][3])
    # Compact sizing when few pathways: smaller row height, less white space
    if n_pathways <= 4:
        row_h = 0.6 + n_pathways * 0.45
    else:
        row_h = 1.2 + min(10, max(4, n_pathways * 0.35))
    total_h = max(4, n_rows * row_h)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(8 if n_pathways <= 5 else 10, total_h),
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    col_real, col_pred = "#2166ac", "#b2182b"
    offset = 0.15

    for ax, (ct, real_df, pred_df, raw_terms) in zip(axes, panels):
        display_terms = [_clean_term(t) for t in raw_terms]
        rows = []
        for t in raw_terms:
            for src_df, src_label in [(real_df, label_real), (pred_df, label_pred)]:
                match = src_df[src_df["Term"] == t]
                if not match.empty:
                    rows.append({"Term": t, "Source": src_label, "NES": float(match.iloc[0]["NES"])})
                else:
                    rows.append({"Term": t, "Source": src_label, "NES": 0.0})
        plot_df = pd.DataFrame(rows)

        for src_label, color, dy in [(label_real, col_real, -offset), (label_pred, col_pred, offset)]:
            sub = plot_df[plot_df["Source"] == src_label]
            y_pos = np.array([raw_terms.index(t) for t in sub["Term"]]) + dy
            sizes = np.clip(np.abs(sub["NES"].values) * 60, 30, 350)
            ax.scatter(
                sub["NES"].values, y_pos, s=sizes, c=color, edgecolors="white",
                linewidths=0.5, alpha=0.85, label=src_label, zorder=3,
            )

        ax.set_yticks(range(len(display_terms)))
        ax.set_yticklabels(display_terms, fontsize=7)
        ax.axvline(0, color="gray", lw=0.5)
        ax.invert_yaxis()
        ax.set_ylabel(ct.replace("_", " "), fontsize=9, rotation=0, labelpad=50)
        ax.tick_params(axis="x", labelbottom=False)

    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("Normalised Enrichment Score (NES)")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    axes[0].set_title(f"GSEA {level}: {dataset} — {label_real} vs {label_pred}\n(top {top_n_pathways} agreed enriched by FDR)")
    plt.subplots_adjust(hspace=0.25)
    plt.tight_layout(pad=0.6, h_pad=0.2)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def generate_nes_scatters(outdir):
    """Generate NES scatter plots from all available GSEA results CSVs."""
    print("\n=== NES Scatter Plots ===")
    for csv_name, label, comparison in [
        ("gsea_level2_results.csv", "L2", "atlas real vs {ds} predicted"),
        ("gsea_level3_results.csv", "L3", "{ds} real vs {ds} predicted"),
        ("gsea_level1_results.csv", "L1", "atlas real vs CV predicted"),
    ]:
        csv_path = outdir / csv_name
        nes_scatter_from_csv(csv_path, label, outdir, comparison=comparison)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None,
                        help="Which comparison level to run (default: all available)")
    parser.add_argument("--nes_scatter", action="store_true",
                        help="Generate NES scatter plots from existing results CSVs")
    parser.add_argument("--joint_dotplot", action="store_true",
                        help="Generate joint GSEA dotplots (pathways x cell types) per dataset")
    parser.add_argument("--stacked_compare", action="store_true",
                        help="Generate stacked GSEA comparison dotplots (cell types stacked, top agreed pathways)")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.stacked_compare:
        generate_stacked_comparison_dotplots(OUTDIR)
        print("\nStacked comparison dotplot generation complete.")
        return
    if args.joint_dotplot:
        generate_joint_dotplots(OUTDIR)
        print("\nJoint dotplot generation complete.")
        return
    if args.nes_scatter:
        generate_nes_scatters(OUTDIR)
        print("\nNES scatter generation complete.")
        return

    if args.level is None or args.level == 3:
        level3_external_real_vs_pred(OUTDIR)
    if args.level is None or args.level == 2:
        level2_atlas_vs_external_pred(OUTDIR)
    if args.level is None or args.level == 1:
        level1_atlas_vs_cv(OUTDIR)

    print("\nStep 8 (GSEA) complete.")


if __name__ == "__main__":
    main()
