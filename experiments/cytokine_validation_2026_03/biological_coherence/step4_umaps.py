#!/usr/bin/env python3
"""
Step 4: Run UMAPs per task (atlas, cv_predictions, or external datasets).

Each task runs as a separate SLURM job. Uses manual matplotlib scatter to avoid
scanpy rasterized conflict. Submit via submit_step4.sh (scavenger_l4).

Usage:
    uv run python step4_umaps.py --task atlas
    uv run python step4_umaps.py --task cv_predictions
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
import scanpy as sc
from scipy.sparse import diags, issparse

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
OUTDIR = BASE / "biological_coherence"
UMAP_DIR = OUTDIR / "plots" / "umaps"
ATLAS_PATH = Path("/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad")
CV_BASE = BASE / "cross_validation"
CV_FOLDS = [CV_BASE / f"run_2026_02_05_fold{i}/eval_last.ckpt/adata_pred.h5ad" for i in range(1, 10)]

EXTERNAL_DATASETS = {
    "GSE96583": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/GSE96583_batch2_IFNbeta_aligned_avg.h5ad"),
        "pred_h5ad": BASE / "run_2026_02_11_all/predictions/GSE96583_batch2_IFNbeta_aligned_avg_predictions.h5ad",
        "clustered_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/clustering/GSE96583_batch2_IFNbeta_clustered_manual_annotated.h5ad"),
        "pert_col": "cytokine",
        "ct_col": "cell_type",
    },
    "GSE178429": {
        "real_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg.h5ad"),
        "pred_h5ad": BASE / "run_2026_02_11_all/predictions/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_predictions.h5ad",
        "clustered_h5ad": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/clustering/GSE178429_PBMCs_stim_scRNAseq_clustered_manual_annotated.h5ad"),
        "pert_col": "cytokine",
        "ct_col": "cell_type",
    },
}


def _to_dense(X):
    if issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


# Atlas/CV ~10M cells: downsample for viz. External: use all cells.
ATLAS_CV_MAX_CELLS = 50000
EXTERNAL_MAX_CELLS = None  # no subsampling

def _run_umap_pipeline(adata: ad.AnnData, max_cells: int | None = 12000, n_top_genes: int = 2000) -> ad.AnnData:
    """Normalize, HVGs, PCA, neighbors, UMAP. Subsample if max_cells set. Sanitize inf/nan for pred data."""
    adata = adata.copy()
    if max_cells is not None and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
    # Sanitize inf/nan (pred data may have overflow)
    X = adata.X
    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X).copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = 0
        adata.X = X
    X_sample = X[: min(200, adata.n_obs)]
    is_counts = (X_sample >= 0).all() and np.allclose(X_sample, X_sample.astype(int))
    if is_counts:
        # Manual normalize_total (avoids scanpy UnboundLocalError with some sparse matrices)
        X = adata.X
        if issparse(X):
            X = X.tocsr()
            row_sums = np.asarray(X.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            adata.X = diags(1e4 / row_sums) @ X
        else:
            row_sums = np.asarray(adata.X.sum(axis=1, keepdims=True))
            row_sums[row_sums == 0] = 1
            adata.X = adata.X / row_sums * 1e4
        sc.pp.log1p(adata)
    else:
        x = _to_dense(adata.X)
        # Clip to 1e4 so log1p stays in safe range (avoids HVG expm1 overflow)
        adata.X = np.clip(x, 0, 1e4)
        sc.pp.log1p(adata)  # put pred data in log space for HVG
    # Sanitize any inf/nan from overflow before HVG
    X = _to_dense(adata.X)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = 0
        adata.X = X
    n_hvg = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata, min_dist=0.3)
    return adata


def _scatter_umap_manual(ax, adata: ad.AnnData, color_col: str, title: str, size: float = 4, alpha: float = 0.6):
    """Plot UMAP using manual scatter (avoids sc.pl.umap rasterized bug)."""
    coords = adata.obsm["X_umap"]
    labels = adata.obs[color_col].astype(str).values
    uniq = pd.Series(labels).dropna().unique()
    uniq = sorted([u for u in uniq if u != "nan"])
    cmap = plt.cm.tab20
    colors = {u: cmap(i % 20) for i, u in enumerate(uniq)}
    c = np.array([colors.get(str(l), (0.5, 0.5, 0.5, alpha)) for l in labels])
    ax.scatter(coords[:, 0], coords[:, 1], c=c, s=size, alpha=alpha)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_aspect("equal")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[u], markersize=6, label=str(u)) for u in uniq[:12]]
    if len(uniq) > 12:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label=f"+{len(uniq)-12} more"))
    if handles:
        ax.legend(handles=handles, fontsize=5, loc="center left", bbox_to_anchor=(1, 0.5))


def _resolve_pert_col(adata: ad.AnnData, pert_col: str) -> str | None:
    """Return actual obs column for perturbation (cytokine/StimType). GSE178429 uses StimType."""
    if pert_col in adata.obs.columns:
        return pert_col
    if "StimType" in adata.obs.columns:
        return "StimType"
    return None


def _save_umaps(adata: ad.AnnData, ds_name: str, color_cols: list[str] | list[tuple[str, str]], outdir: Path, dot_size: float = 4) -> None:
    """color_cols: str uses col for data and title; (data_col, display_label) uses data_col, displays display_label."""
    for item in color_cols:
        if isinstance(item, tuple):
            data_col, display = item
        else:
            data_col, display = item, item
        if data_col not in adata.obs.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        _scatter_umap_manual(ax, adata, data_col, f"{ds_name} - {display}", size=dot_size)
        plt.tight_layout()
        fname = outdir / f"umap_{ds_name.replace(' ', '_')}_{display.replace(' ', '_')}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname.name}")


def _ensure_control_treated(adata: ad.AnnData, control_label: str = "control") -> str:
    """Ensure control vs treated column; return its name. control_label='PBS' for atlas/CV."""
    if "treatment" in adata.obs.columns:
        return "treatment"
    for col in ["cytokine", "condition", "stim"]:
        if col in adata.obs.columns:
            vals = adata.obs[col].astype(str).str.lower()
            ctrl = vals.isin(["control", "untreated", "none", "", "pbs"])
            adata.obs["control_vs_treated"] = np.where(ctrl, control_label, "treated")
            return "control_vs_treated"
    return ""

def run_atlas() -> None:
    """Downsampled UMAP: cell type + treatment vs control only. No other outputs stored."""
    print("=== Atlas UMAP (downsampled, cell type + treatment) ===")
    if not ATLAS_PATH.exists():
        print(f"  [SKIP] Atlas not found: {ATLAS_PATH}")
        return
    adata = ad.read_h5ad(ATLAS_PATH)
    adata = _run_umap_pipeline(adata, max_cells=ATLAS_CV_MAX_CELLS)
    color_cols = ["cell_type"]
    pert_col = _ensure_control_treated(adata, control_label="PBS")
    if pert_col:
        color_cols.append(pert_col)
    _save_umaps(adata, "cytokine_atlas", color_cols, UMAP_DIR)
    del adata
    print("Atlas UMAP done.")


def run_cv_predictions() -> None:
    """Downsampled UMAP: cell type + treatment vs control. Nothing stored, only PNGs."""
    print("=== CV Predictions UMAP (downsampled, cell type + treatment vs control) ===")
    cv_adatas = []
    for fp in CV_FOLDS:
        if fp.exists():
            cv_adatas.append(ad.read_h5ad(fp))
    if not cv_adatas:
        print("  [SKIP] No CV fold h5ad files found.")
        return
    adata = ad.concat(cv_adatas, join="inner")
    del cv_adatas
    adata.obs_names_make_unique()
    adata = _run_umap_pipeline(adata, max_cells=ATLAS_CV_MAX_CELLS)
    color_cols = ["cell_type"]
    pert_col = _ensure_control_treated(adata, control_label="PBS")
    if pert_col:
        color_cols.append(pert_col)
    _save_umaps(adata, "cv_predictions", color_cols, UMAP_DIR, dot_size=2)
    del adata
    print("CV Predictions UMAP done.")


def run_external_task(ds_name: str, task: str) -> None:
    ds_info = EXTERNAL_DATASETS[ds_name]
    if task == "real":
        path = ds_info["clustered_h5ad"] if ds_info["clustered_h5ad"].exists() else ds_info["real_h5ad"]
        if not path.exists():
            print(f"  [SKIP] {ds_name} real: {path} not found")
            return
        adata = ad.read_h5ad(path)
        adata = _run_umap_pipeline(adata, max_cells=None)  # external: all cells
        pert_resolved = _resolve_pert_col(adata, ds_info["pert_col"])
        cols = [ds_info["ct_col"], (pert_resolved, "cytokine")] if pert_resolved else [ds_info["ct_col"]]
        _save_umaps(adata, f"{ds_name}_real", cols, UMAP_DIR, dot_size=6)
    elif task == "pred":
        path = ds_info["pred_h5ad"]
        if not path.exists():
            print(f"  [SKIP] {ds_name} pred: {path} not found")
            return
        adata = ad.read_h5ad(path)
        adata = _run_umap_pipeline(adata, max_cells=None)  # external: all cells
        pert_resolved = _resolve_pert_col(adata, ds_info["pert_col"])
        cols = [ds_info["ct_col"], (pert_resolved, "cytokine")] if pert_resolved else [ds_info["ct_col"]]
        _save_umaps(adata, f"{ds_name}_pred", cols, UMAP_DIR, dot_size=6)
    elif task == "joint":
        real_path = ds_info["real_h5ad"]
        pred_path = ds_info["pred_h5ad"]
        if not real_path.exists() or not pred_path.exists():
            print(f"  [SKIP] {ds_name} joint: missing real or pred")
            return
        adata_real = ad.read_h5ad(real_path)
        adata_pred = ad.read_h5ad(pred_path)
        if "is_imputed" in adata_real.var.columns:
            keep = ~adata_real.var["is_imputed"].astype(bool).values
            adata_real = adata_real[:, keep].copy()
            adata_pred = adata_pred[:, keep].copy()
        adata_real.obs["source"] = "real"
        adata_pred.obs["source"] = "predicted"
        pert_resolved = _resolve_pert_col(adata_real, ds_info["pert_col"])
        copy_cols = [ds_info["ct_col"]] + ([pert_resolved] if pert_resolved else [])
        for col in copy_cols:
            if col in adata_real.obs.columns and col not in adata_pred.obs.columns:
                adata_pred.obs[col] = adata_real.obs[col].values
        combined = ad.concat([adata_real, adata_pred], join="inner")
        combined = _run_umap_pipeline(combined, max_cells=None)  # external: all cells
        joint_cols = [ds_info["ct_col"]] + ([(pert_resolved, "cytokine")] if pert_resolved else []) + ["source"]
        _save_umaps(combined, f"{ds_name}_real_vs_pred", joint_cols, UMAP_DIR, dot_size=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="atlas | cv_predictions | GSE96583_real | GSE96583_pred | GSE96583_joint | GSE178429_real | GSE178429_pred | GSE178429_joint")
    args = parser.parse_args()
    task = args.task

    UMAP_DIR.mkdir(parents=True, exist_ok=True)

    if task == "atlas":
        run_atlas()
    elif task == "cv_predictions":
        run_cv_predictions()
    elif task.startswith("GSE96583_") or task.startswith("GSE178429_"):
        parts = task.rsplit("_", 1)
        if len(parts) == 2:
            ds_name, subtask = parts
            if ds_name in EXTERNAL_DATASETS and subtask in ("real", "pred", "joint"):
                run_external_task(ds_name, subtask)
            else:
                raise SystemExit(f"Unknown task: {task}")
        else:
            raise SystemExit(f"Unknown task: {task}")
    else:
        raise SystemExit(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
