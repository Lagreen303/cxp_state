#!/usr/bin/env python3
"""
Generate CV heatmaps and box plots with n cells per cell type.

Creates:
  - heatmap_pr_auc.png, heatmap_pearson_delta.png (with n cells in axis labels)
  - plots/celltype_box_pr_auc.png, plots/celltype_box_pearson_delta.png
  - plots/cytokine_box_pr_auc.png, plots/cytokine_box_pearson_delta.png

Requires: cv_celltype_results_long.csv in this directory.
Atlas report for n cells: /iridisfs/ddnb/Luke/data/parse-10M-90cytokines/parse_VCC_report.txt

Usage:
    cd /iridisfs/ddnb/Luke/scFMs/state
    uv run python experiments/cytokine_validation_2026/cross_validation/aggregated_celltype_results/make_cv_heatmaps_and_boxes.py
"""
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_DIR = Path("/iridisfs/ddnb/Luke/scFMs/state")
MAKE_HEATMAP = STATE_DIR / "analyse_results" / "make_eval_heatmap.py"
INPUT_CSV = SCRIPT_DIR / "cv_celltype_results_long.csv"
ATLAS_REPORT = Path("/iridisfs/ddnb/Luke/data/parse-10M-90cytokines/parse_VCC_report.txt")

if __name__ == "__main__":
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Run aggregate_cv_celltypes.py first.")
        sys.exit(1)
    cmd = [
        sys.executable, str(MAKE_HEATMAP),
        "--input", str(INPUT_CSV),
        "--outdir", str(SCRIPT_DIR),
        "--metrics", "pearson_delta", "pr_auc",
        "--atlas-report", str(ATLAS_REPORT),
        "--boxplots",
    ]
    subprocess.run(cmd, check=True)
    print("Done. Heatmaps in aggregated_celltype_results/, box plots in aggregated_celltype_results/plots/")
