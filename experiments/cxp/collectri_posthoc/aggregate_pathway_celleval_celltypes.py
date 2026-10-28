#!/usr/bin/env python3
"""
Aggregate per-cell-type *_results.csv from CollecTRI pathway cell-eval.

Looks for:
  <eval_dir>/pathway_celleval/<CellType>_results.csv

Writes:
  pathway_celleval/aggregated_celltype_results/collectri_celltype_results_long.csv
  pathway_celleval/aggregated_celltype_results/collectri_<sanitised_cell_type>_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

STATE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STATE_ROOT / "experiments" / "cxp"))

from cxp_collectri_posthoc import (  # noqa: E402
    eval_dir_for_step,
    load_posthoc_config,
    resolve_latest_step,
)

RESULTS_SUFFIX = "_results.csv"
AGG_SUFFIX = "_agg_results.csv"
OUTPUT_SUBDIR = "aggregated_celltype_results"
LONG_CSV_NAME = "collectri_celltype_results_long.csv"


def collect_results(pathway_celleval_dir: Path, checkpoint_step: int) -> pd.DataFrame:
    if not pathway_celleval_dir.is_dir():
        raise FileNotFoundError(pathway_celleval_dir)

    result_files = sorted(
        p
        for p in pathway_celleval_dir.glob(f"*{RESULTS_SUFFIX}")
        if not p.name.endswith(AGG_SUFFIX)
    )
    if not result_files:
        raise RuntimeError(f"No *{RESULTS_SUFFIX} (non-agg) in {pathway_celleval_dir}")

    rows = []
    for path in result_files:
        cell_type = path.stem.removesuffix("_results")
        df = pd.read_csv(path)
        if "perturbation" not in df.columns:
            continue
        df = df.copy()
        df["train_step"] = checkpoint_step
        df["cell_type"] = cell_type
        rows.append(df)

    if not rows:
        raise RuntimeError("No per-perturbation results files found.")

    combined = pd.concat(rows, ignore_index=True)
    meta_cols = ["train_step", "cell_type", "perturbation"]
    other_cols = [c for c in combined.columns if c not in meta_cols]
    return combined[meta_cols + other_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(STATE_ROOT / "experiments/cxp/configs/cxp_collectri_posthoc.yaml"),
    )
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--pathway-celleval-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_posthoc_config(args.config)
    run_dir = Path(cfg["run_dir"])
    step = args.checkpoint_step if args.checkpoint_step is not None else resolve_latest_step(run_dir)

    if args.pathway_celleval_dir:
        celleval_dir = Path(args.pathway_celleval_dir)
    else:
        suffix = str(cfg.get("eval_dir_suffix", ""))
        celleval_dir = eval_dir_for_step(run_dir, step, suffix=suffix) / "pathway_celleval"

    combined = collect_results(celleval_dir, step)
    out_dir = celleval_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_long = out_dir / LONG_CSV_NAME
    combined.to_csv(out_long, index=False)
    print(f"[DONE] Wrote long-form results to {out_long} ({len(combined)} rows)")

    for cell_type, sub in combined.groupby("cell_type"):
        safe_name = cell_type.replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"collectri_{safe_name}_results.csv"
        sub.to_csv(out_path, index=False)
        print(f"  - Wrote {cell_type!r} results to {out_path}")


if __name__ == "__main__":
    main()
