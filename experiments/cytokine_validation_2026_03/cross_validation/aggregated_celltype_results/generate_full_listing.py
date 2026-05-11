#!/usr/bin/env python3
"""
Generate a full listing of everything in the heatmaps and all plots.
Output: PLOT_FULL_LISTING.md + heatmap_pr_auc_full.csv + heatmap_pearson_delta_full.csv
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LONG_CSV = BASE_DIR / "cv_celltype_results_long.csv"
OUT_MD = BASE_DIR / "PLOT_FULL_LISTING.md"


def _fmt(x: float, dp: int = 3) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.{dp}f}"


def main() -> None:
    df = pd.read_csv(LONG_CSV)
    grouped = (
        df.groupby(["cell_type", "perturbation"], as_index=False)
        .agg({"pr_auc": "mean", "pearson_delta": "mean"})
    )

    cell_types = sorted(grouped["cell_type"].unique().tolist())
    cytokines = sorted(grouped["perturbation"].unique().tolist())

    # Pivot matrices (cytokine = rows, cell_type = cols)
    pr_auc_mat = grouped.pivot(index="perturbation", columns="cell_type", values="pr_auc")
    pearson_mat = grouped.pivot(index="perturbation", columns="cell_type", values="pearson_delta")

    # Reorder to match cytokines order
    pr_auc_mat = pr_auc_mat.reindex(index=cytokines, columns=cell_types)
    pearson_mat = pearson_mat.reindex(index=cytokines, columns=cell_types)

    # Save full matrices as CSV
    pr_auc_mat.to_csv(BASE_DIR / "heatmap_pr_auc_full.csv")
    pearson_mat.to_csv(BASE_DIR / "heatmap_pearson_delta_full.csv")

    # Build markdown
    lines = [
        "# Full listing: heatmaps and plots",
        "",
        "## 1. All plots",
        "",
        "| Plot | File | Description |",
        "|------|------|-------------|",
        "| PR-AUC heatmap | `heatmap_pr_auc.png` | Cytokine × cell type, mean PR-AUC over folds |",
        "| Pearson delta heatmap | `heatmap_pearson_delta.png` | Cytokine × cell type, mean pearson_delta over folds |",
        "| PR-AUC heatmap (clustered) | `heatmap_pr_auc_clustered.png` | Same with hierarchical clustering dendrograms |",
        "| Pearson delta heatmap (clustered) | `heatmap_pearson_delta_clustered.png` | Same with hierarchical clustering dendrograms |",
        "| PR-AUC box plot | `plots/celltype_box_pr_auc.png` | PR-AUC distribution per cell type |",
        "| Pearson delta box plot | `plots/celltype_box_pearson_delta.png` | Pearson delta distribution per cell type |",
        "| Cytokine PR-AUC box plot | `plots/cytokine_box_pr_auc.png` | PR-AUC distribution per cytokine |",
        "| Cytokine Pearson delta box plot | `plots/cytokine_box_pearson_delta.png` | Pearson delta distribution per cytokine |",
        "| Scatter: pearson vs PR-AUC | `plots/scatter_pearson_vs_pr_auc.png` | Each point = cell type × cytokine |",
        "| Cell type cytokine ranking | `plots/celltype_rank_pr_auc_<celltype>.png` | One per cell type (17 files) |",
        "",
        "---",
        "",
        "## 2. Cell types (17)",
        "",
    ]
    for ct in cell_types:
        lines.append(f"- {ct}")
    lines.extend(["", "---", "", "## 3. Cytokines (90)", ""])
    for cy in cytokines:
        lines.append(f"- {cy}")
    lines.extend([
        "",
        "---",
        "",
        "## 4. Heatmap matrices (mean over 9 folds)",
        "",
        "Full numeric data saved to:",
        "- `heatmap_pr_auc_full.csv` — rows = cytokines, columns = cell types",
        "- `heatmap_pearson_delta_full.csv` — rows = cytokines, columns = cell types",
        "",
        "### 4a. PR-AUC heatmap — per cytokine × cell type",
        "",
    ])
    # Table: cytokine | cell_type1 | cell_type2 | ...
    header = "| Cytokine | " + " | ".join(ct.replace(" ", "&nbsp;") for ct in cell_types) + " |"
    sep = "|" + "---|" * (len(cell_types) + 1)
    lines.append(header)
    lines.append(sep)
    for cy in cytokines:
        row = f"| {cy} |"
        for ct in cell_types:
            val = pr_auc_mat.loc[cy, ct] if cy in pr_auc_mat.index and ct in pr_auc_mat.columns else None
            row += f" {_fmt(val)} |"
        lines.append(row)
    lines.extend([
        "",
        "### 4b. Pearson delta heatmap — per cytokine × cell type",
        "",
    ])
    lines.append(header)
    lines.append(sep)
    for cy in cytokines:
        row = f"| {cy} |"
        for ct in cell_types:
            val = pearson_mat.loc[cy, ct] if cy in pearson_mat.index and ct in pearson_mat.columns else None
            row += f" {_fmt(val)} |"
        lines.append(row)
    # === Section 5: Box plot contents with exact stats ===
    pr_auc_col = "pr_auc"
    pearson_col = "pearson_delta"
    if "pr_auc_mean" in grouped.columns:
        pr_auc_col = "pr_auc_mean"
        pearson_col = "pearson_delta_mean"

    def _box_stats(series):
        q = series.quantile([0.25, 0.5, 0.75])
        return {
            "n": series.count(),
            "mean": series.mean(),
            "median": q[0.5],
            "sd": series.std(),
            "q1": q[0.25],
            "q3": q[0.75],
            "iqr": q[0.75] - q[0.25],
            "min": series.min(),
            "max": series.max(),
        }

    # Per-cell-type PR-AUC (celltype_box_pr_auc.png)
    ct_pr = grouped.groupby("cell_type")[pr_auc_col]
    ct_pr_stats = {ct: _box_stats(ct_pr.get_group(ct)) for ct in cell_types}
    ct_pr_order = sorted(cell_types, key=lambda c: ct_pr_stats[c]["median"], reverse=True)

    # Per-cell-type Pearson delta (celltype_box_pearson_delta.png)
    ct_pearson = grouped.groupby("cell_type")[pearson_col]
    ct_pearson_stats = {ct: _box_stats(ct_pearson.get_group(ct)) for ct in cell_types}
    ct_pearson_order = sorted(cell_types, key=lambda c: ct_pearson_stats[c]["median"], reverse=True)

    # Per-cytokine PR-AUC (cytokine_box_pr_auc.png)
    cy_pr = grouped.groupby("perturbation")[pr_auc_col]
    cy_pr_stats = {cy: _box_stats(cy_pr.get_group(cy)) for cy in cytokines}
    cy_pr_order = sorted(cytokines, key=lambda c: cy_pr_stats[c]["median"], reverse=True)

    # Per-cytokine Pearson delta (cytokine_box_pearson_delta.png)
    cy_pearson = grouped.groupby("perturbation")[pearson_col]
    cy_pearson_stats = {cy: _box_stats(cy_pearson.get_group(cy)) for cy in cytokines}
    cy_pearson_order = sorted(cytokines, key=lambda c: cy_pearson_stats[c]["median"], reverse=True)

    lines.extend([
        "",
        "---",
        "",
        "## 5. Box plot contents",
        "",
        "### 5a. Cell type PR-AUC box plot (`plots/celltype_box_pr_auc.png`)",
        "",
        f"Each cell type: n = {len(cytokines)} cytokines. Ordered by median PR-AUC (descending).",
        "",
        "| Cell type | n | Mean | Median | SD | Q1 | Q3 | IQR | Min | Max |",
        "|-----------|---|------|--------|-----|-----|-----|-----|-----|-----|",
    ])
    for ct in ct_pr_order:
        s = ct_pr_stats[ct]
        lines.append(f"| {ct} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['sd'])} | {_fmt(s['q1'])} | {_fmt(s['q3'])} | {_fmt(s['iqr'])} | {_fmt(s['min'])} | {_fmt(s['max'])} |")

    lines.extend([
        "",
        "### 5b. Cell type Pearson delta box plot (`plots/celltype_box_pearson_delta.png`)",
        "",
        f"Each cell type: n = {len(cytokines)} cytokines. Ordered by median pearson_delta (descending).",
        "",
        "| Cell type | n | Mean | Median | SD | Q1 | Q3 | IQR | Min | Max |",
        "|-----------|---|------|--------|-----|-----|-----|-----|-----|-----|",
    ])
    for ct in ct_pearson_order:
        s = ct_pearson_stats[ct]
        lines.append(f"| {ct} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['sd'])} | {_fmt(s['q1'])} | {_fmt(s['q3'])} | {_fmt(s['iqr'])} | {_fmt(s['min'])} | {_fmt(s['max'])} |")

    lines.extend([
        "",
        "### 5c. Cytokine PR-AUC box plot (`plots/cytokine_box_pr_auc.png`)",
        "",
        f"Each cytokine: n = {len(cell_types)} cell types. Ordered by median PR-AUC (descending).",
        "",
        "| Cytokine | n | Mean | Median | SD | Q1 | Q3 | IQR | Min | Max |",
        "|----------|---|------|--------|-----|-----|-----|-----|-----|-----|",
    ])
    for cy in cy_pr_order:
        s = cy_pr_stats[cy]
        lines.append(f"| {cy} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['sd'])} | {_fmt(s['q1'])} | {_fmt(s['q3'])} | {_fmt(s['iqr'])} | {_fmt(s['min'])} | {_fmt(s['max'])} |")

    lines.extend([
        "",
        "### 5d. Cytokine Pearson delta box plot (`plots/cytokine_box_pearson_delta.png`)",
        "",
        f"Each cytokine: n = {len(cell_types)} cell types. Ordered by median pearson_delta (descending).",
        "",
        "| Cytokine | n | Mean | Median | SD | Q1 | Q3 | IQR | Min | Max |",
        "|----------|---|------|--------|-----|-----|-----|-----|-----|-----|",
    ])
    for cy in cy_pearson_order:
        s = cy_pearson_stats[cy]
        lines.append(f"| {cy} | {s['n']} | {_fmt(s['mean'])} | {_fmt(s['median'])} | {_fmt(s['sd'])} | {_fmt(s['q1'])} | {_fmt(s['q3'])} | {_fmt(s['iqr'])} | {_fmt(s['min'])} | {_fmt(s['max'])} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Scatter plot (E)",
        "",
        f"Each of the {len(grouped)} points represents one cell type × cytokine condition (mean over folds).",
        "Labelled: top 10 by combined score (pearson_delta + pr_auc) / 2.",
        "",
    ])

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote heatmap_pr_auc_full.csv")
    print(f"Wrote heatmap_pearson_delta_full.csv")


if __name__ == "__main__":
    main()
