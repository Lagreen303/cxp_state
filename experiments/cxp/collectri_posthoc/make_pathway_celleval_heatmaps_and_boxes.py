#!/usr/bin/env python3
"""
Generate pathway cell-eval heatmaps and box plots (CollecTRI posthoc).

Creates under pathway_celleval/aggregated_celltype_results/:
  - heatmap_pr_auc.png, heatmap_pearson_delta.png
  - plots/celltype_box_*.png, plots/cytokine_box_*.png

Requires collectri_celltype_results_long.csv (run aggregate_pathway_celleval_celltypes.py first).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STATE_ROOT = Path(__file__).resolve().parents[3]
MAKE_HEATMAP = STATE_ROOT / "analyse_results" / "make_eval_heatmap.py"
ATLAS_REPORT = Path("/iridisfs/ddnb/Luke/data/parse-10M-90cytokines/parse_VCC_report.txt")
LONG_CSV_NAME = "collectri_celltype_results_long.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregated-dir",
        type=str,
        required=True,
        help="pathway_celleval/aggregated_celltype_results",
    )
    parser.add_argument(
        "--atlas-report",
        type=str,
        default=str(ATLAS_REPORT),
    )
    args = parser.parse_args()

    agg_dir = Path(args.aggregated_dir)
    input_csv = agg_dir / LONG_CSV_NAME
    if not input_csv.is_file():
        print(f"Error: {input_csv} not found. Run aggregate_pathway_celleval_celltypes.py first.")
        sys.exit(1)

    eval_ckpt_dir = agg_dir.parent.parent
    counts_h5ad = eval_ckpt_dir / "adata_pred.h5ad"

    cmd = [
        sys.executable,
        str(MAKE_HEATMAP),
        "--input",
        str(input_csv),
        "--outdir",
        str(agg_dir),
        "--metrics",
        "pearson_delta",
        "pr_auc",
        "--boxplots",
    ]
    if counts_h5ad.is_file():
        cmd.extend(["--counts-h5ad", str(counts_h5ad), "--counts-obs-col", "cell_type"])
    else:
        cmd.extend(["--atlas-report", str(args.atlas_report)])
    subprocess.run(cmd, check=True)
    print(f"Done. Heatmaps in {agg_dir}/, box plots in {agg_dir}/plots/")


if __name__ == "__main__":
    main()
