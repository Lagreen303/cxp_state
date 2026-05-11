#!/usr/bin/env python3
"""
Step 5: Consolidate outputs for thesis.

- Build a figure index CSV mapping filenames to captions.
- Print a summary of all quantitative metrics.

Run after steps 1-4 have completed.

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/biological_coherence/step5_consolidate.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUTDIR = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/biological_coherence")
PLOTS = OUTDIR / "plots"


def build_figure_index():
    """Scan all plot directories and build a CSV index of figures."""
    rows = []
    for subdir in sorted(PLOTS.iterdir()):
        if not subdir.is_dir():
            continue
        category = subdir.name
        for f in sorted(subdir.glob("*.png")):
            parts = f.stem.split("_")
            rows.append({
                "category": category,
                "filename": f.name,
                "path": str(f.relative_to(OUTDIR)),
                "caption": "",
            })

    df = pd.DataFrame(rows)
    out = OUTDIR / "figure_index.csv"
    df.to_csv(out, index=False)
    print(f"Figure index: {len(df)} figures -> {out}")
    return df


def summarise_coherence():
    """Print and re-save a clean summary of gene coherence metrics."""
    path = OUTDIR / "gene_coherence_metrics.csv"
    if not path.exists():
        print("gene_coherence_metrics.csv not found (step 2 not complete yet).")
        return

    df = pd.read_csv(path)
    print(f"\n=== Gene coherence metrics ({len(df)} rows) ===")

    summary = df.groupby("dataset").agg(
        n_comparisons=("cell_type", "count"),
        mean_spearman_logFC=("spearman_logFC", "mean"),
        mean_pearson_logFC=("pearson_logFC", "mean"),
        mean_jaccard_de=("jaccard_de", "mean"),
        mean_precision_de=("precision_de", "mean"),
        mean_recall_de=("recall_de", "mean"),
    ).round(3)
    print(summary.to_string())

    summary_path = OUTDIR / "gene_coherence_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")

    # Per cell-type breakdown
    ct_summary = df.groupby(["dataset", "cell_type"]).agg(
        spearman_logFC=("spearman_logFC", "mean"),
        pearson_logFC=("pearson_logFC", "mean"),
        jaccard_de=("jaccard_de", "mean"),
        n_de_atlas=("n_de_atlas", "mean"),
        n_de_pred=("n_de_pred", "mean"),
    ).round(3)
    ct_path = OUTDIR / "gene_coherence_by_celltype.csv"
    ct_summary.to_csv(ct_path)
    print(f"Per cell-type breakdown saved to {ct_path}")


def main():
    print("=" * 60)
    print("Step 5: Consolidating outputs for thesis")
    print("=" * 60)

    fig_index = build_figure_index()
    print(f"\nFigure categories:")
    print(fig_index["category"].value_counts().to_string())

    summarise_coherence()

    # List all output files
    print(f"\n=== All output files in {OUTDIR} ===")
    for f in sorted(OUTDIR.glob("*.csv")):
        print(f"  {f.name}")
    for subdir in sorted(PLOTS.iterdir()):
        if subdir.is_dir():
            n = len(list(subdir.glob("*.png")))
            print(f"  plots/{subdir.name}/: {n} figures")

    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
