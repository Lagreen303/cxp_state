#!/usr/bin/env python3
"""
Minimal atlas and CV UMAP: downsampled, coloured by cell type and treatment vs control only.
Produces 4 PNGs total. Nothing else stored.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
OUTDIR = BASE / "biological_coherence"
UMAP_DIR = OUTDIR / "plots" / "umaps"
ATLAS_PATH = Path("/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad")
CV_BASE = BASE / "cross_validation"
CV_FOLDS = [CV_BASE / f"run_2026_02_05_fold{i}/eval_last.ckpt/adata_pred.h5ad" for i in range(1, 10)]

N_CELLS = 50000  # atlas/CV from ~10M: downsampled for viz


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def _run_umap(adata: ad.AnnData, sanitize: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Subsample, normalize, HVG, PCA, neighbors, UMAP. Returns (coords, obs) only.
    sanitize=True for pred data (inf/nan fix); False for atlas."""
    adata = adata.copy()
    if adata.n_obs > N_CELLS:
        sc.pp.subsample(adata, n_obs=N_CELLS, random_state=42)
    if sanitize:
        X = _to_dense(adata.X)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = 0
            adata.X = X
    X_sample = _to_dense(adata.X[: min(200, adata.n_obs)])
    is_counts = (X_sample >= 0).all() and np.allclose(X_sample, X_sample.astype(int))
    if is_counts:
        from scipy.sparse import diags
        X = adata.X
        if issparse(X):
            X = X.tocsr()
            row_sums = np.asarray(X.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            adata.X = diags(1e4 / row_sums) @ X
        else:
            row_sums = np.asarray(X.sum(axis=1, keepdims=True))
            row_sums[row_sums == 0] = 1
            adata.X = X / row_sums * 1e4
        sc.pp.log1p(adata)
    else:
        adata.X = np.clip(_to_dense(adata.X), 0, 1e6)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata, min_dist=0.3)
    coords = np.asarray(adata.obsm["X_umap"]).copy()
    obs = adata.obs.copy()
    del adata
    return coords, obs


def _ensure_control_vs_treated(obs: pd.DataFrame) -> pd.Series:
    """Return series: control vs treated."""
    if "treatment" in obs.columns:
        col = obs["treatment"].astype(str)
    elif "cytokine" in obs.columns:
        col = obs["cytokine"].astype(str)
    else:
        return pd.Series(["unknown"] * len(obs), index=obs.index)
    ctrl = col.str.lower().isin(["control", "untreated", "none", "nan", ""])
    return pd.Series(np.where(ctrl, "control", "treated"), index=obs.index)


def _plot_umap(coords: np.ndarray, labels: pd.Series, title: str, outpath: Path) -> None:
    """Scatter plot coloured by labels."""
    uniq = sorted(labels.dropna().astype(str).unique())
    uniq = [u for u in uniq if u != "nan"]
    cmap = plt.cm.tab20
    colors = {u: cmap(i % 20) for i, u in enumerate(uniq)}
    c = np.array([colors.get(str(l), (0.5, 0.5, 0.5, 0.6)) for l in labels])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=c, s=4, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_aspect("equal")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[u], markersize=6, label=u) for u in uniq[:15]]
    if len(uniq) > 15:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label=f"+{len(uniq)-15}"))
    if handles:
        ax.legend(handles=handles, fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def run_atlas() -> None:
    print("=== Atlas UMAP (downsampled, cell type + control vs treated) ===")
    if not ATLAS_PATH.exists():
        print(f"  [SKIP] {ATLAS_PATH} not found")
        return
    adata = ad.read_h5ad(ATLAS_PATH)
    coords, obs = _run_umap(adata)
    del adata
    ct_col = "cell_type" if "cell_type" in obs.columns else obs.columns[0]
    obs["control_vs_treated"] = _ensure_control_vs_treated(obs)
    _plot_umap(coords, obs[ct_col], "Atlas — cell type", UMAP_DIR / "umap_atlas_cell_type.png")
    _plot_umap(coords, obs["control_vs_treated"], "Atlas — control vs treated", UMAP_DIR / "umap_atlas_control_vs_treated.png")
    print("Atlas done.")


def run_cv() -> None:
    print("=== CV Predictions UMAP (downsampled, cell type + control vs treated) ===")
    adatas = []
    for fp in CV_FOLDS:
        if fp.exists():
            adatas.append(ad.read_h5ad(fp))
    if not adatas:
        print("  [SKIP] No CV folds found")
        return
    adata = ad.concat(adatas, join="inner")
    del adatas
    adata.obs_names_make_unique()
    coords, obs = _run_umap(adata)
    del adata
    ct_col = "cell_type" if "cell_type" in obs.columns else None
    if ct_col is None:
        candidates = [c for c in obs.columns if "cell" in str(c).lower()]
        ct_col = candidates[0] if candidates else obs.columns[0]
    obs["control_vs_treated"] = _ensure_control_vs_treated(obs)
    _plot_umap(coords, obs[ct_col], "CV predictions — cell type", UMAP_DIR / "umap_cv_cell_type.png")
    _plot_umap(coords, obs["control_vs_treated"], "CV predictions — control vs treated", UMAP_DIR / "umap_cv_control_vs_treated.png")
    print("CV done.")


def main() -> None:
    UMAP_DIR.mkdir(parents=True, exist_ok=True)
    run_atlas()
    run_cv()
    print("\nDone. 4 PNGs in plots/umaps/.")


if __name__ == "__main__":
    main()
