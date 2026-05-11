#!/usr/bin/env python3
"""
Generate a summary statistics file for CV validation plots (A–E).
Use these exact numbers when writing about the figures.

Output: PLOT_SUMMARY_STATS.md
"""

from pathlib import Path

import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
LONG_CSV = BASE_DIR / "cv_celltype_results_long.csv"
OUT_MD = BASE_DIR / "PLOT_SUMMARY_STATS.md"


def _fmt(x: float, dp: int = 3) -> str:
    return f"{x:.{dp}f}"


def main() -> None:
    df = pd.read_csv(LONG_CSV)
    grouped = (
        df.groupby(["cell_type", "perturbation"], as_index=False)
        .agg({"pr_auc": "mean", "pearson_delta": "mean"})
        .rename(columns={"pr_auc": "pr_auc_mean", "pearson_delta": "pearson_delta_mean"})
    )

    n_folds = df["fold"].nunique()
    n_cell_types = grouped["cell_type"].nunique()
    n_cytokines = grouped["perturbation"].nunique()
    n_conditions = len(grouped)

    # A) PR-AUC heatmap stats
    pr_auc = grouped["pr_auc_mean"]
    pr_auc_overall = {
        "mean": pr_auc.mean(),
        "median": pr_auc.median(),
        "std": pr_auc.std(),
        "min": pr_auc.min(),
        "max": pr_auc.max(),
        "q25": pr_auc.quantile(0.25),
        "q75": pr_auc.quantile(0.75),
    }

    pr_auc_by_ct = grouped.groupby("cell_type")["pr_auc_mean"].agg(["mean", "median", "std", "min", "max"])
    pr_auc_best_ct = pr_auc_by_ct["median"].idxmax()
    pr_auc_worst_ct = pr_auc_by_ct["median"].idxmin()

    pr_auc_by_cyt = grouped.groupby("perturbation")["pr_auc_mean"].agg(["mean", "median", "std", "min", "max"])
    pr_auc_best_cyt = pr_auc_by_cyt["median"].idxmax()
    pr_auc_worst_cyt = pr_auc_by_cyt["median"].idxmin()

    # B) Pearson delta heatmap stats
    pearson = grouped["pearson_delta_mean"]
    pearson_overall = {
        "mean": pearson.mean(),
        "median": pearson.median(),
        "std": pearson.std(),
        "min": pearson.min(),
        "max": pearson.max(),
        "q25": pearson.quantile(0.25),
        "q75": pearson.quantile(0.75),
    }

    pearson_by_ct = grouped.groupby("cell_type")["pearson_delta_mean"].agg(["mean", "median", "std", "min", "max"])
    pearson_best_ct = pearson_by_ct["median"].idxmax()
    pearson_worst_ct = pearson_by_ct["median"].idxmin()

    pearson_by_cyt = grouped.groupby("perturbation")["pearson_delta_mean"].agg(["mean", "median", "std", "min", "max"])
    pearson_best_cyt = pearson_by_cyt["median"].idxmax()
    pearson_worst_cyt = pearson_by_cyt["median"].idxmin()

    # C & D) Box plot stats (same data, per cell type) - IQR computed in loops below

    # E) Scatter: pearson_delta vs pr_auc
    valid = grouped[["pearson_delta_mean", "pr_auc_mean"]].dropna()
    if len(valid) >= 3:
        r_pearson, p_pearson = stats.pearsonr(valid["pearson_delta_mean"], valid["pr_auc_mean"])
        r_spearman, p_spearman = stats.spearmanr(valid["pearson_delta_mean"], valid["pr_auc_mean"])
    else:
        r_pearson, p_pearson = float("nan"), float("nan")
        r_spearman, p_spearman = float("nan"), float("nan")

    top10 = grouped.copy()
    top10["combined"] = (top10["pearson_delta_mean"] + top10["pr_auc_mean"]) / 2
    top10 = top10.sort_values("combined", ascending=False).head(10)
    top10_list = [f"{r['cell_type']} – {r['perturbation']} (pr_auc={_fmt(r['pr_auc_mean'])}, r={_fmt(r['pearson_delta_mean'])})" for _, r in top10.iterrows()]

    lines = [
        "# CV Validation Plot Summary Statistics",
        "",
        "Use these exact numbers when writing about Figures A–E. Data: 9-fold leave-one-cytokine-out CV, mean per (cell type × cytokine) across folds.",
        "",
        "---",
        "",
        "## Data overview",
        f"- **Folds**: {n_folds}",
        f"- **Cell types**: {n_cell_types}",
        f"- **Cytokines**: {n_cytokines}",
        f"- **Conditions** (cell type × cytokine): {n_conditions}",
        "",
        "---",
        "",
        "## A) PR-AUC heatmap",
        "",
        "### Overall",
        f"- Mean: {_fmt(pr_auc_overall['mean'])}",
        f"- Median: {_fmt(pr_auc_overall['median'])}",
        f"- SD: {_fmt(pr_auc_overall['std'])}",
        f"- Range: [{_fmt(pr_auc_overall['min'])}–{_fmt(pr_auc_overall['max'])}]",
        f"- IQR (Q25–Q75): {_fmt(pr_auc_overall['q25'])}–{_fmt(pr_auc_overall['q75'])}",
        "",
        "### Best / worst cell type (by median PR-AUC)",
        f"- Best: **{pr_auc_best_ct}** (median = {_fmt(pr_auc_by_ct.loc[pr_auc_best_ct, 'median'])})",
        f"- Worst: **{pr_auc_worst_ct}** (median = {_fmt(pr_auc_by_ct.loc[pr_auc_worst_ct, 'median'])})",
        "",
        "### Best / worst cytokine (by median PR-AUC)",
        f"- Best: **{pr_auc_best_cyt}** (median = {_fmt(pr_auc_by_cyt.loc[pr_auc_best_cyt, 'median'])})",
        f"- Worst: **{pr_auc_worst_cyt}** (median = {_fmt(pr_auc_by_cyt.loc[pr_auc_worst_cyt, 'median'])})",
        "",
        "---",
        "",
        "## B) Pearson delta heatmap",
        "",
        "### Overall",
        f"- Mean: {_fmt(pearson_overall['mean'])}",
        f"- Median: {_fmt(pearson_overall['median'])}",
        f"- SD: {_fmt(pearson_overall['std'])}",
        f"- Range: [{_fmt(pearson_overall['min'])}–{_fmt(pearson_overall['max'])}]",
        f"- IQR (Q25–Q75): {_fmt(pearson_overall['q25'])}–{_fmt(pearson_overall['q75'])}",
        "",
        "### Best / worst cell type (by median pearson_delta)",
        f"- Best: **{pearson_best_ct}** (median = {_fmt(pearson_by_ct.loc[pearson_best_ct, 'median'])})",
        f"- Worst: **{pearson_worst_ct}** (median = {_fmt(pearson_by_ct.loc[pearson_worst_ct, 'median'])})",
        "",
        "### Best / worst cytokine (by median pearson_delta)",
        f"- Best: **{pearson_best_cyt}** (median = {_fmt(pearson_by_cyt.loc[pearson_best_cyt, 'median'])})",
        f"- Worst: **{pearson_worst_cyt}** (median = {_fmt(pearson_by_cyt.loc[pearson_worst_cyt, 'median'])})",
        "",
        "---",
        "",
        "## C) PR-AUC box plot (by cell type)",
        "",
        "Same statistics as A, displayed as box plot per cell type. Cell types ordered by median PR-AUC (descending).",
        "",
        "### Per-cell-type medians (for quick reference)",
        "",
    ]

    for ct in pr_auc_by_ct.sort_values("median", ascending=False).index:
        med = pr_auc_by_ct.loc[ct, "median"]
        iqr = grouped.groupby("cell_type")["pr_auc_mean"].get_group(ct).quantile(0.75) - grouped.groupby("cell_type")["pr_auc_mean"].get_group(ct).quantile(0.25)
        lines.append(f"- {ct}: median = {_fmt(med)}, IQR = {_fmt(iqr)}")

    lines.extend([
        "",
        "---",
        "",
        "## D) Pearson delta box plot (by cell type)",
        "",
        "Same statistics as B, displayed as box plot per cell type. Cell types ordered by median pearson_delta (descending).",
        "",
        "### Per-cell-type medians (for quick reference)",
        "",
    ])

    for ct in pearson_by_ct.sort_values("median", ascending=False).index:
        med = pearson_by_ct.loc[ct, "median"]
        iqr = grouped.groupby("cell_type")["pearson_delta_mean"].get_group(ct).quantile(0.75) - grouped.groupby("cell_type")["pearson_delta_mean"].get_group(ct).quantile(0.25)
        lines.append(f"- {ct}: median = {_fmt(med)}, IQR = {_fmt(iqr)}")

    lines.extend([
        "",
        "---",
        "",
        "## E) Pearson delta vs PR-AUC scatter",
        "",
        "### Correlation between metrics",
        f"- Pearson r = {_fmt(r_pearson)} (p {('< 0.001' if p_pearson < 0.001 else f'= {_fmt(p_pearson)}')})",
        f"- Spearman ρ = {_fmt(r_spearman)} (p {('< 0.001' if p_spearman < 0.001 else f'= {_fmt(p_spearman)}')})",
        "",
        "### Top 10 conditions (by combined score = (pearson_delta + pr_auc) / 2)",
        "",
    ])

    for i, item in enumerate(top10_list, 1):
        lines.append(f"{i}. {item}")

    lines.extend([
        "",
        "---",
        "",
        "## Figure → file mapping",
        "| Panel | Plot | File |",
        "|-------|------|------|",
        "| A | PR-AUC heatmap | `heatmap_pr_auc.png` |",
        "| B | Pearson delta heatmap | `heatmap_pearson_delta.png` |",
        "| C | PR-AUC box plot | `plots/celltype_box_pr_auc.png` |",
        "| D | Pearson delta box plot | `plots/celltype_box_pearson_delta.png` |",
        "| E | pearson_delta vs pr_auc scatter | `plots/scatter_pearson_vs_pr_auc.png` |",
        "",
    ])

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
