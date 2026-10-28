#!/usr/bin/env python3
"""Rough comparison: nested CxP pathway cell-eval vs 2026_03 CV gene metrics (shared cytokines)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CELLX_ROOT = Path("/iridisfs/ddnb/Luke/cellxpathway")
STATE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CELLX_ROOT))

from cellxpathway.eval import plots as cxp_plots  # noqa: E402

DEFAULT_HOLDOUT_PERTS = ("IFN-gamma", "IFN-lambda1", "IL-17A", "IL-6", "TNF-alpha")
DEFAULT_SEEN5_PERTS = ("GM-CSF", "IL-2", "IL-10", "G-CSF", "IL-7")
METRICS = ("pearson_delta", "pr_auc")

DEFAULT_CXP_LONG = (
    STATE_ROOT
    / "experiments/cxp/CollecTRI/run_cxp_v2_collectri/eval_step=50000.ckpt/pathway_celleval"
    / "aggregated_celltype_results/collectri_celltype_results_long.csv"
)
DEFAULT_CV_LONG = (
    STATE_ROOT
    / "experiments/cytokine_validation_2026_03/cross_validation/aggregated_celltype_results"
    / "cv_celltype_results_long.csv"
)
DEFAULT_OUT = STATE_ROOT / "experiments/cxp/figures/tmp"


def _plot_style(perts_mode: str, cv_label: str | None) -> dict[str, str]:
    """File stems and figure copy keyed by --perts (holdout vs training-seen seen5)."""
    if perts_mode == "seen5":
        return {
            "paired_table": "cxp_vs_cv_seen5_train_paired_table.csv",
            "bars": "cxp_vs_cv_seen5_train_metrics_bars",
            "scatter": "cxp_vs_cv_seen5_train_metrics_scatter",
            "heatmap": "cxp_vs_cv_seen5_train_heatmap",
            "per_celltype": "cxp_vs_cv_seen5_train_per_celltype_bars",
            "per_cytokine": "cxp_vs_cv_seen5_train_per_cytokine_bars",
            "cohort": "training-seen cytokines",
            "cxp_panel": "CxP pathway",
            "cv_panel": cv_label or "CV gene",
        }
    return {
        "paired_table": "cxp_vs_cv_holdout_paired_table.csv",
        "bars": "cxp_vs_cv_holdout_metrics_bars",
        "scatter": "cxp_vs_cv_holdout_metrics_scatter",
        "heatmap": "cxp_vs_cv_holdout_heatmap",
        "per_celltype": "cxp_vs_cv_holdout_per_celltype_bars",
        "per_cytokine": "cxp_vs_cv_holdout_per_cytokine_bars",
        "cohort": "held-out cytokines",
        "cxp_panel": "CxP pathway (50k)",
        "cv_panel": cv_label or "CV gene (9-fold mean)",
    }


def _load_paired(cxp_path: Path, cv_path: Path, perts: tuple[str, ...]) -> pd.DataFrame:
    dc = pd.read_csv(cxp_path)
    dv = pd.read_csv(cv_path)
    dc = dc[dc["perturbation"].isin(perts)]
    dv = dv[dv["perturbation"].isin(perts)]
    cv_mean = dv.groupby(["cell_type", "perturbation"], as_index=False)[list(METRICS)].mean()
    cxp_mean = dc.groupby(["cell_type", "perturbation"], as_index=False)[list(METRICS)].mean()
    m = cxp_mean.merge(
        cv_mean,
        on=["cell_type", "perturbation"],
        suffixes=("_cxp_pathway", "_cv_gene"),
        how="inner",
    )
    if m.empty:
        raise RuntimeError("No overlapping (cell_type, perturbation) rows between CxP and CV.")
    return m


def _pivot_metric(paired: pd.DataFrame, metric: str, suffix: str, perts: tuple[str, ...]) -> pd.DataFrame:
    col = f"{metric}_{suffix}"
    mat = paired.pivot(index="perturbation", columns="cell_type", values=col)
    return mat.reindex(index=list(perts))


def _ordered_cell_types(paired: pd.DataFrame) -> list[str]:
    return sorted(paired["cell_type"].unique(), key=str)


def plot_heatmap_triptych(
    paired: pd.DataFrame,
    out_dir: Path,
    metric: str,
    perts: tuple[str, ...],
    style: dict[str, str],
) -> None:
    """Cytokine × cell_type heatmaps: CxP, CV, and CxP−CV."""
    cell_order = _ordered_cell_types(paired)
    cxp = _pivot_metric(paired, metric, "cxp_pathway", perts)[cell_order]
    cv = _pivot_metric(paired, metric, "cv_gene", perts)[cell_order]
    diff = cxp - cv

    vlim = np.nanmax(np.abs(np.concatenate([cxp.to_numpy(), cv.to_numpy()])))
    vlim = max(vlim, 1e-6)
    dlim = np.nanmax(np.abs(diff.to_numpy()))
    dlim = max(dlim, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    panels = [
        (cxp, style["cxp_panel"], "rocket_r", (0, vlim)),
        (cv, style["cv_panel"], "rocket_r", (0, vlim)),
        (diff, "CxP − CV", "RdBu_r", (-dlim, dlim)),
    ]
    for ax, (mat, title, cmap, (vmin, vmax)) in zip(axes, panels, strict=True):
        sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0 if cmap == "RdBu_r" else None,
            linewidths=0.3,
            linecolor="0.85",
            cbar_kws={"label": metric},
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Cell type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.suptitle(f"{metric}: {style['cohort']} × cell types", fontsize=10, y=1.03)
    fig.tight_layout()
    cxp_plots.save_figure(
        fig,
        f"{style['heatmap']}_{metric}",
        out_dir,
        description=f"CxP vs CV heatmap triptych for {metric}",
        flat=True,
    )


def plot_per_celltype_bars(paired: pd.DataFrame, out_dir: Path, stem: str, perts: tuple[str, ...]) -> None:
    """One panel per cell type: cytokines on x, CxP vs CV for both metrics (2 rows)."""
    hues = cxp_plots.CATEGORICAL_HUES
    cell_order = _ordered_cell_types(paired)
    n_ct = len(cell_order)
    ncols = 4
    nrows = int(np.ceil(n_ct / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(3.4 * ncols, 2.2 * nrows * 2), squeeze=False)

    x = np.arange(len(perts))
    width = 0.36
    for idx, ct in enumerate(cell_order):
        col = idx % ncols
        block = idx // ncols
        sub = paired[paired["cell_type"] == ct]
        for mi, metric in enumerate(METRICS):
            ax = axes[block * 2 + mi, col]
            cxp_col = f"{metric}_cxp_pathway"
            cv_col = f"{metric}_cv_gene"
            vals_cxp = sub.set_index("perturbation").reindex(perts)[cxp_col]
            vals_cv = sub.set_index("perturbation").reindex(perts)[cv_col]
            ax.bar(x - width / 2, vals_cxp, width, color=hues[0], label="CxP")
            ax.bar(x + width / 2, vals_cv, width, color=hues[1], label="CV")
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(perts, rotation=55, ha="right", fontsize=6)
            if mi == 0:
                ax.set_title(ct, fontsize=8)
            if col == 0:
                ax.set_ylabel(metric, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, axis="y", alpha=0.2)

    for j in range(n_ct, nrows * ncols):
        col = j % ncols
        block = j // ncols
        for mi in range(2):
            axes[block * 2 + mi, col].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Per cell type: CxP pathway vs CV gene by cytokine", fontsize=10, y=1.02)
    fig.tight_layout()
    cxp_plots.save_figure(
        fig,
        stem,
        out_dir,
        description="Per cell type grouped bar comparison",
        flat=True,
    )


def plot_per_cytokine_bars(paired: pd.DataFrame, out_dir: Path, stem: str, perts: tuple[str, ...]) -> None:
    """One panel per cytokine: cell types on x, CxP vs CV (2 metric rows)."""
    hues = cxp_plots.CATEGORICAL_HUES
    cell_order = _ordered_cell_types(paired)
    n_ct = len(cell_order)
    fig, axes = plt.subplots(2, len(perts), figsize=(3.5 * len(perts), 7), squeeze=False)
    x = np.arange(n_ct)
    width = 0.36

    for j, pert in enumerate(perts):
        sub = paired[paired["perturbation"] == pert].set_index("cell_type").reindex(cell_order)
        for mi, metric in enumerate(METRICS):
            ax = axes[mi, j]
            cxp_col = f"{metric}_cxp_pathway"
            cv_col = f"{metric}_cv_gene"
            ax.bar(x - width / 2, sub[cxp_col], width, color=hues[0], label="CxP")
            ax.bar(x + width / 2, sub[cv_col], width, color=hues[1], label="CV")
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(cell_order, rotation=65, ha="right", fontsize=6)
            if mi == 0:
                ax.set_title(pert, fontsize=9)
            if j == 0:
                ax.set_ylabel(metric)
            ax.grid(True, axis="y", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Per cytokine: CxP pathway vs CV gene by cell type", fontsize=10, y=1.02)
    fig.tight_layout()
    cxp_plots.save_figure(
        fig,
        stem,
        out_dir,
        description="Per cytokine grouped bar comparison",
        flat=True,
    )


def plot_grouped_means(
    paired: pd.DataFrame,
    out_dir: Path,
    stem: str,
    perts: tuple[str, ...],
    style: dict[str, str],
) -> None:
    hues = cxp_plots.CATEGORICAL_HUES
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    x = np.arange(len(perts))
    width = 0.36

    for ax, metric in zip(axes, METRICS, strict=True):
        cxp_col = f"{metric}_cxp_pathway"
        cv_col = f"{metric}_cv_gene"
        cxp_m = paired.groupby("perturbation")[cxp_col].mean().reindex(perts)
        cv_m = paired.groupby("perturbation")[cv_col].mean().reindex(perts)
        cxp_s = paired.groupby("perturbation")[cxp_col].std().reindex(perts)
        cv_s = paired.groupby("perturbation")[cv_col].std().reindex(perts)

        ax.bar(x - width / 2, cxp_m, width, yerr=cxp_s, color=hues[0], label=style["cxp_panel"], capsize=3)
        ax.bar(x + width / 2, cv_m, width, yerr=cv_s, color=hues[1], label=style["cv_panel"], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(perts, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        f"{style['cohort'].title()}: mean over shared cell types (± SD across types)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    cxp_plots.save_figure(
        fig,
        stem,
        out_dir,
        description="CxP pathway vs CV gene metrics by perturbation (bar)",
        flat=True,
    )


def plot_paired_scatter(paired: pd.DataFrame, out_dir: Path, stem: str, perts: tuple[str, ...]) -> None:
    n = len(perts)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.5), squeeze=False)
    hues = cxp_plots.CATEGORICAL_HUES

    for j, pert in enumerate(perts):
        sub = paired[paired["perturbation"] == pert]
        for i, metric in enumerate(METRICS):
            ax = axes[i, j]
            x = sub[f"{metric}_cv_gene"].to_numpy()
            y = sub[f"{metric}_cxp_pathway"].to_numpy()
            ax.scatter(x, y, s=22, color=hues[0], alpha=0.85, edgecolors="none")
            lo = float(np.nanmin([x.min(), y.min(), -0.1]))
            hi = float(np.nanmax([x.max(), y.max(), 0.1]))
            pad = 0.05 * (hi - lo + 1e-6)
            lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, color="0.55", lw=0.8, zorder=0)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            if i == 0:
                ax.set_title(pert, fontsize=9)
            if j == 0:
                ax.set_ylabel(metric)
            if i == 1:
                ax.set_xlabel("CV gene")
            else:
                ax.set_xlabel("")

    fig.suptitle("Per cell type: CV gene (x) vs CxP pathway (y)", fontsize=10, y=1.02)
    fig.tight_layout()
    cxp_plots.save_figure(
        fig,
        stem,
        out_dir,
        description="CxP pathway vs CV gene paired scatter by perturbation",
        flat=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cxp-long", type=Path, default=DEFAULT_CXP_LONG)
    p.add_argument("--cv-long", type=Path, default=DEFAULT_CV_LONG)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--perts",
        type=str,
        default="holdout",
        choices=("holdout", "seen5"),
        help="holdout=test1 cytokines; seen5=training-seen GM-CSF/IL-2/IL-10/G-CSF/IL-7",
    )
    p.add_argument(
        "--cv-label",
        type=str,
        default=None,
        help="Legend/panel label for CV side (default depends on --perts).",
    )
    args = p.parse_args()

    perts = DEFAULT_HOLDOUT_PERTS if args.perts == "holdout" else DEFAULT_SEEN5_PERTS
    style = _plot_style(args.perts, args.cv_label)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cxp_plots.apply_style(CELLX_ROOT)

    paired = _load_paired(args.cxp_long.resolve(), args.cv_long.resolve(), perts)
    summary_path = out_dir / style["paired_table"]
    paired.to_csv(summary_path, index=False)

    plot_grouped_means(paired, out_dir, style["bars"], perts, style)
    plot_paired_scatter(paired, out_dir, style["scatter"], perts)
    for metric in METRICS:
        plot_heatmap_triptych(paired, out_dir, metric, perts, style)
    plot_per_celltype_bars(paired, out_dir, style["per_celltype"], perts)
    plot_per_cytokine_bars(paired, out_dir, style["per_cytokine"], perts)
    print(f"[INFO] Wrote figures and {summary_path} under {out_dir}")


if __name__ == "__main__":
    main()
