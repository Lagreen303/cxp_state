#!/usr/bin/env python3
"""
Step 6: Combined coherence heatmaps — 3 versions:
  1. Both datasets in one figure
  2. GSE96583 only
  3. GSE178429 only

Each version shows 4 metrics (Spearman logFC, Jaccard DE, Precision DE, Recall DE)
as subplots with cell types on x-axis. Cell type labels include n cells when available.

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step6_combined_heatmap.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/biological_coherence")
METRICS_CSV = BASE / "gene_coherence_metrics.csv"
OUTDIR = BASE / "plots" / "heatmaps"
REPORTS_DIR = BASE / "reports"

METRICS = [
    ("spearman_logFC", "Spearman logFC", (-1, 1), "RdYlGn"),
    ("jaccard_de", "Jaccard DE overlap", (0, 1), "YlOrRd"),
    ("precision_de", "Precision DE", (0, 1), "YlOrRd"),
    ("recall_de", "Recall DE", (0, 1), "YlOrRd"),
]

DATASET_LABELS = {
    "GSE96583": "GSE96583 (IFN-\u03b2)",
    "GSE178429": "GSE178429 (IFN-\u03b3)",
}


def load_n_cells_per_cell_type() -> dict[str, dict[str, int]]:
    """Load n cells per (dataset, cell_type) from reports. Returns {dataset: {cell_type: n}}."""
    result: dict[str, dict[str, int]] = {}
    for ds in ("GSE96583", "GSE178429"):
        path = REPORTS_DIR / f"cell_counts_{ds}.txt"
        if not path.exists():
            result[ds] = {}
            continue
        text = path.read_text(encoding="utf-8")
        in_section = False
        result[ds] = {}
        for line in text.splitlines():
            if "--- By cell type ---" in line:
                in_section = True
                continue
            if in_section:
                stripped = line.strip()
                if stripped.startswith("---") or stripped.startswith("Total:"):
                    break
                m = re.match(r"^(.+?):\s+([\d,]+)$", stripped)
                if m and not m.group(1).strip().lower().startswith("total"):
                    ct = m.group(1).strip()
                    n = int(m.group(2).replace(",", ""))
                    result[ds][ct] = n
    return result


def make_heatmap_figure(
    df: pd.DataFrame,
    datasets: list[str],
    title_suffix: str,
    outpath: Path,
    n_cells: dict[str, dict[str, int]] | None = None,
):
    n_cells = n_cells or {}
    n_metrics = len(METRICS)
    n_rows = len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_metrics,
        figsize=(4 * n_metrics, 1.2 * n_rows + 1.8),
        squeeze=False,
    )

    for col_idx, (metric, label, (vmin, vmax), cmap) in enumerate(METRICS):
        for row_idx, ds in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            sub = df[df["dataset"] == ds].copy()
            sub = sub.sort_values("cell_type")

            pivot = sub.set_index("cell_type")[[metric]].T
            pivot = pivot.fillna(0)

            sns.heatmap(
                pivot, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                cbar=col_idx == n_metrics - 1,
                cbar_kws={"shrink": 0.8} if col_idx == n_metrics - 1 else {},
                yticklabels=[DATASET_LABELS.get(ds, ds)],
                xticklabels=pivot.columns if row_idx == n_rows - 1 else [],
            )
            if row_idx == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")

            if row_idx == n_rows - 1:
                col_names = list(pivot.columns)
                if n_cells.get(ds):
                    col_names = [f"{ct}\n(n={n_cells[ds].get(ct, 0):,})" for ct in col_names]
                ax.set_xticks(range(len(col_names)))
                ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=6)

    fig.suptitle(f"Gene-level coherence metrics{title_suffix}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath.name}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_CSV)
    n_cells = load_n_cells_per_cell_type()

    # Version 1: Combined
    make_heatmap_figure(
        df, ["GSE96583", "GSE178429"],
        " — both datasets",
        OUTDIR / "heatmap_combined_all_metrics.png",
        n_cells=n_cells,
    )

    # Version 2: GSE96583 only
    make_heatmap_figure(
        df[df["dataset"] == "GSE96583"],
        ["GSE96583"],
        " — GSE96583 (IFN-\u03b2)",
        OUTDIR / "heatmap_GSE96583_all_metrics.png",
        n_cells=n_cells,
    )

    # Version 3: GSE178429 only
    make_heatmap_figure(
        df[df["dataset"] == "GSE178429"],
        ["GSE178429"],
        " — GSE178429 (IFN-\u03b3)",
        OUTDIR / "heatmap_GSE178429_all_metrics.png",
        n_cells=n_cells,
    )

    print("Step 6 complete.")


if __name__ == "__main__":
    main()
