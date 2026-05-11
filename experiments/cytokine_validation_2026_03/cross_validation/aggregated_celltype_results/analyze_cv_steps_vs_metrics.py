#!/usr/bin/env python3
"""
Correlate 2026_03 CV eval metrics (eval_last.ckpt aggregated CSVs) with training depth
(global_step in checkpoints/last.ckpt per fold).

Merges atlas cytokine cell counts (backed AnnData, cached to cv_steps_analysis/atlas_cytokine_counts.csv).

Outputs under ./cv_steps_analysis/ include:
  fold_training_depth.csv, fold_level_metrics.csv, perturbation_fold_metrics.csv,
  perturbation_fold_metrics_with_atlas.csv (if atlas), fold_atlas_abundance_covariate.csv,
  correlations.txt,
  fig_*.png (including fig_pearson_vs_atlas_abundance.png, fig_residualized_logstep_vs_pearson_delta.png).

Run:
  cd /iridisfs/ddnb/Luke/scFMs/state
  uv run python experiments/cytokine_validation_2026_03/cross_validation/aggregated_celltype_results/analyze_cv_steps_vs_metrics.py
  uv run python .../analyze_cv_steps_vs_metrics.py --no-atlas
  uv run python .../analyze_cv_steps_vs_metrics.py --use-atlas-cache   # CSV only, no h5ad
"""

from __future__ import annotations

import argparse
import datetime as _dt
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
RUN_PREFIX = "run_2026_02_05_fold"
LONG_CSV = SCRIPT_DIR / "cv_celltype_results_long.csv"
DEFAULT_OUTDIR = SCRIPT_DIR / f"cv_steps_analysis_{_dt.date.today().isoformat()}"

DEFAULT_ATLAS_H5AD = (
    "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad"
)

METRICS = ["pr_auc", "pearson_delta", "roc_auc", "de_spearman_sig"]


def read_last_global_steps() -> pd.DataFrame:
    rows = []
    for k in range(1, 10):
        ckpt = CV_DIR / f"{RUN_PREFIX}{k}" / "checkpoints" / "last.ckpt"
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        rows.append(
            {
                "fold": k,
                "global_step": int(obj.get("global_step", -1)),
                "epoch": int(obj.get("epoch", -1)) if obj.get("epoch") is not None else -1,
                "last_ckpt": str(ckpt),
            }
        )
    return pd.DataFrame(rows)


def load_cytokine_counts(
    atlas_path: Path,
    cache_csv: Path,
    use_cache_only: bool,
) -> pd.Series | None:
    """Series index=cytokine name, value=n_cells in atlas."""
    if use_cache_only:
        if not cache_csv.is_file():
            return None
        df = pd.read_csv(cache_csv)
        if "cytokine" not in df.columns or "n_cells" not in df.columns:
            return None
        return pd.Series(df["n_cells"].values, index=df["cytokine"].astype(str))

    if cache_csv.is_file():
        df = pd.read_csv(cache_csv)
        if "cytokine" in df.columns and "n_cells" in df.columns:
            return pd.Series(df["n_cells"].values, index=df["cytokine"].astype(str))

    if not atlas_path.is_file():
        return None
    import anndata as ad

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adat = ad.read_h5ad(atlas_path, backed="r")
        vc = adat.obs["cytokine"].astype(str).value_counts()
    adat.file.close()
    pd.DataFrame({"cytokine": vc.index.astype(str), "n_cells": vc.values}).to_csv(
        cache_csv, index=False
    )
    return vc


def fig_residualized_step(
    pert_df: pd.DataFrame,
    out_path: Path,
    has_atlas: bool,
) -> None:
    """Pearson_delta vs log_step after removing linear effect of log1p atlas (if available)."""
    from sklearn.linear_model import LinearRegression

    y = pert_df["mean_pearson_delta"].astype(float).values
    log_step = np.log(pert_df["global_step"].astype(float).values)
    if has_atlas:
        z = np.log1p(pert_df["atlas_n_cells"].astype(float).values).reshape(-1, 1)
        y_r = y - LinearRegression().fit(z, y).predict(z).ravel()
        x_r = log_step - LinearRegression().fit(z, log_step).predict(z).ravel()
        xlab = "log(global_step) resid | log1p(atlas n_cells)"
        ylab = "mean_pearson_delta resid | log1p(atlas n_cells)"
    else:
        y_r, x_r, xlab, ylab = y, log_step, "log(global_step)", "mean_pearson_delta"

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(x_r, y_r, c=pert_df["fold"], cmap="tab10", alpha=0.85)
    for i, row in enumerate(pert_df.itertuples()):
        ax.annotate(
            str(row.fold),
            (x_r[i], y_r[i]),
            textcoords="offset points",
            xytext=(3, 2),
            fontsize=7,
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("Residualized association (perturbation means)")
    fig.colorbar(sc, ax=ax, label="fold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-atlas",
        action="store_true",
        help="Skip AnnData read (no atlas merge or atlas figures).",
    )
    ap.add_argument(
        "--atlas-h5ad",
        type=Path,
        default=Path(DEFAULT_ATLAS_H5AD),
        help="Path to atlas h5ad for cytokine cell counts.",
    )
    ap.add_argument(
        "--use-atlas-cache",
        action="store_true",
        help="Use cached atlas counts only (no h5ad read).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory (default: date-stamped folder).",
    )
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_csv = outdir / "atlas_cytokine_counts.csv"

    df = pd.read_csv(LONG_CSV)
    steps = read_last_global_steps()
    steps.to_csv(outdir / "fold_training_depth.csv", index=False)

    merged = df.merge(steps, on="fold", how="left")

    # ---- Fold-level means
    fold_stats = []
    for fold, sub in merged.groupby("fold"):
        row = {"fold": int(fold), "global_step": int(sub["global_step"].iloc[0])}
        for m in METRICS:
            if m in sub.columns:
                row[f"mean_{m}"] = sub[m].mean()
                row[f"median_{m}"] = sub[m].median()
                row[f"std_{m}"] = sub[m].std(ddof=1)
        row["n_eval_rows"] = len(sub)
        fold_stats.append(row)
    fold_df = pd.DataFrame(fold_stats).sort_values("fold")
    fold_df.to_csv(outdir / "fold_level_metrics.csv", index=False)

    # ---- Per (fold, perturbation)
    pert_rows = []
    for (fold, pert), sub in merged.groupby(["fold", "perturbation"]):
        r = {
            "fold": int(fold),
            "perturbation": pert,
            "global_step": int(sub["global_step"].iloc[0]),
            "n_cell_types": len(sub),
        }
        for m in METRICS:
            if m in sub.columns:
                r[f"mean_{m}"] = sub[m].mean()
        pert_rows.append(r)
    pert_df = pd.DataFrame(pert_rows)
    pert_df.to_csv(outdir / "perturbation_fold_metrics.csv", index=False)

    has_atlas = not args.no_atlas
    atlas_ser: pd.Series | None = None
    if has_atlas:
        atlas_ser = load_cytokine_counts(
            args.atlas_h5ad, cache_csv, use_cache_only=args.use_atlas_cache
        )
        if atlas_ser is None:
            has_atlas = False
            warnings.warn("Atlas counts unavailable; continuing without atlas covariate.")

    if has_atlas and atlas_ser is not None:
        pert_df["atlas_n_cells"] = pert_df["perturbation"].map(atlas_ser)
        missing = pert_df["atlas_n_cells"].isna().sum()
        if missing:
            warnings.warn(f"{missing} perturbations missing in atlas count table (check names).")
        pert_df["log1p_atlas_n_cells"] = np.log1p(pert_df["atlas_n_cells"].fillna(0))
        pert_df.to_csv(outdir / "perturbation_fold_metrics_with_atlas.csv", index=False)

        # Fold-level mean log abundance of held-out set
        fold_atlas = (
            pert_df.groupby("fold")
            .agg(mean_log1p_atlas_n=("log1p_atlas_n_cells", "mean"))
            .reset_index()
        )
        fold_atlas.to_csv(outdir / "fold_atlas_abundance_covariate.csv", index=False)

    # ---- Correlations
    lines = []
    for name, frame in [("fold_level_n9", fold_df), ("perturbation_level_n90", pert_df)]:
        gs = frame["global_step"].astype(float)
        for m in METRICS:
            col = f"mean_{m}" if f"mean_{m}" in frame.columns else m
            if col not in frame.columns:
                continue
            y = frame[col].astype(float)
            if len(frame) < 3:
                continue
            r = np.corrcoef(gs, y)[0, 1]
            rho, p = spearmanr(gs, y)
            lines.append(f"{name}\t{col}\tpearson_r={r:.4f}\tspearman_rho={rho:.4f}\tp={p:.4g}\n")

    if has_atlas and "log1p_atlas_n_cells" in pert_df.columns:
        for m in METRICS:
            col = f"mean_{m}"
            if col not in pert_df.columns:
                continue
            rho, p = spearmanr(pert_df["log1p_atlas_n_cells"], pert_df[col])
            lines.append(
                f"perturbation_level_vs_atlas\t{col}\tspearman_rho={rho:.4f}\tp={p:.4g}\n"
            )

    (outdir / "correlations.txt").write_text("".join(lines))
    print((outdir / "correlations.txt").read_text())

    # ---- Figures
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m in zip(axes, ["pr_auc", "pearson_delta"]):
        col = f"mean_{m}"
        ax.scatter(fold_df["global_step"], fold_df[col], s=80, c=fold_df["fold"], cmap="tab10")
        for _, r in fold_df.iterrows():
            ax.annotate(
                str(int(r["fold"])),
                (r["global_step"], r[col]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        ax.set_xlabel("global_step (last.ckpt)")
        ax.set_ylabel(f"Fold mean {m}")
        ax.set_title("Fold aggregate vs training steps\n(n=9 folds)")
    fig.tight_layout()
    fig.savefig(outdir / "fig_fold_global_step_vs_fold_means.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc = None
    for ax, m in zip(axes, ["pr_auc", "pearson_delta"]):
        col = f"mean_{m}"
        sc = ax.scatter(
            pert_df["global_step"],
            pert_df[col],
            c=pert_df["fold"],
            cmap="tab10",
            alpha=0.75,
            s=35,
        )
        ax.set_xlabel("global_step (last.ckpt for that fold)")
        ax.set_ylabel(f"Mean {m} across cell types")
        ax.set_title(f"Held-out cytokine eval vs fold training depth\n(n={len(pert_df)} fold×cytokine)")
    if sc is not None:
        fig.colorbar(sc, ax=list(axes), label="fold")
    fig.tight_layout()
    fig.savefig(outdir / "fig_perturbation_mean_vs_global_step.png")
    plt.close(fig)

    med = fold_df["global_step"].median()
    pert_df = pert_df.copy()
    pert_df["step_group"] = np.where(pert_df["global_step"] < med, "below_median_step", "at_or_above_median")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m in zip(axes, ["pr_auc", "pearson_delta"]):
        col = f"mean_{m}"
        data = [pert_df.loc[pert_df["step_group"] == g, col].values for g in ["below_median_step", "at_or_above_median"]]
        ax.boxplot(data, tick_labels=["below median\nfold global_step", "at/above median"])
        ax.set_ylabel(f"Perturbation mean {m}")
        ax.set_title(f"Median global_step split = {med:.0f}")
    fig.suptitle("Per held-out cytokine (mean over cell types)", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig_low_vs_high_step_boxplots.png")
    plt.close(fig)

    low_folds = {4, 7}
    pert_df["fold_group"] = pert_df["fold"].apply(lambda f: "fold_4_7_low_step" if f in low_folds else "other_folds")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, m in zip(axes, ["pr_auc", "pearson_delta"]):
        col = f"mean_{m}"
        data = [pert_df.loc[pert_df["fold_group"] == g, col].values for g in ["fold_4_7_low_step", "other_folds"]]
        ax.boxplot(data, tick_labels=["folds 4&7\n(12k–14k steps)", "other folds"])
        ax.set_ylabel(f"Perturbation mean {m}")
    fig.suptitle("Folds 4 & 7 stopped early vs others", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig_fold47_vs_others_boxplots.png")
    plt.close(fig)

    if has_atlas and "atlas_n_cells" in pert_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(
            np.log1p(pert_df["atlas_n_cells"]),
            pert_df["mean_pearson_delta"],
            c=pert_df["fold"],
            cmap="tab10",
            alpha=0.8,
        )
        ax.set_xlabel("log1p(atlas cells per cytokine)")
        ax.set_ylabel("mean_pearson_delta (per fold×cytokine)")
        fig.colorbar(sc, ax=ax, label="fold")
        fig.tight_layout()
        fig.savefig(outdir / "fig_pearson_vs_atlas_abundance.png")
        plt.close(fig)

        fig_residualized_step(
            pert_df.reset_index(drop=True),
            outdir / "fig_residualized_logstep_vs_pearson_delta.png",
            True,
        )

    print(f"[DONE] Wrote analysis to {outdir}")


if __name__ == "__main__":
    main()
