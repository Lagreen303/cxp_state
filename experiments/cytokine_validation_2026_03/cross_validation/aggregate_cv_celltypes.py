#!/usr/bin/env python3
"""
Aggregate per-cell-type *_results.csv across all CV folds.

For each fold run (run_2026_02_05_foldK), this script looks for:
  run_2026_02_05_foldK/eval_last.ckpt/<CellType>_results.csv

It concatenates them into:
  - aggregated_celltype_results/cv_celltype_results_long.csv
    (all folds, all cell types, all cytokines)

and also writes one CSV per cell type:
  - aggregated_celltype_results/cv_<sanitised_cell_type>_results.csv
"""

from pathlib import Path

import pandas as pd


CV_DIR = Path(__file__).resolve().parent
RUN_PREFIX = "run_2026_02_05_fold"
EVAL_DIR_NAMES = ("eval_last.ckpt", "eval_final.ckpt")
RESULTS_SUFFIX = "_results.csv"
OUTPUT_DIR = CV_DIR / "aggregated_celltype_results"


def find_eval_dir(run_dir: Path) -> Path | None:
    """Return eval subdir under run_dir (preferring eval_last.ckpt)."""
    for name in EVAL_DIR_NAMES:
        d = run_dir / name
        if d.is_dir():
            return d
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_rows = []

    for k in range(1, 10):
        run_dir = CV_DIR / f"{RUN_PREFIX}{k}"
        if not run_dir.is_dir():
            print(f"[SKIP] Fold {k}: run dir {run_dir} not found")
            continue

        eval_dir = find_eval_dir(run_dir)
        if eval_dir is None:
            print(f"[SKIP] Fold {k}: no eval_* directory in {run_dir}")
            continue

        result_files = sorted(p for p in eval_dir.glob(f"*{RESULTS_SUFFIX}"))
        if not result_files:
            print(f"[SKIP] Fold {k}: no *{RESULTS_SUFFIX} files in {eval_dir}")
            continue

        for path in result_files:
            cell_type = path.stem.removesuffix("_results")
            df = pd.read_csv(path)
            if "perturbation" not in df.columns:
                # Not a per-perturbation results file; skip
                continue
            df = df.copy()
            df["fold"] = k
            df["cell_type"] = cell_type
            all_rows.append(df)

        print(f"[OK] Fold {k}: collected {len(result_files)} cell-type results from {eval_dir}")

    if not all_rows:
        raise RuntimeError("No *_results.csv files found across folds.")

    combined = pd.concat(all_rows, ignore_index=True)

    # Put meta columns first
    meta_cols = ["fold", "cell_type", "perturbation"]
    other_cols = [c for c in combined.columns if c not in meta_cols]
    combined = combined[meta_cols + other_cols]

    out_long = OUTPUT_DIR / "cv_celltype_results_long.csv"
    combined.to_csv(out_long, index=False)
    print(f"[DONE] Wrote long-form results to {out_long} ({len(combined)} rows)")

    # Also write one CSV per cell type for convenience
    for cell_type, sub in combined.groupby("cell_type"):
        safe_name = cell_type.replace(" ", "_").replace("/", "_")
        out_path = OUTPUT_DIR / f"cv_{safe_name}_results.csv"
        sub.to_csv(out_path, index=False)
        print(f"  - Wrote {cell_type!r} results to {out_path}")


if __name__ == "__main__":
    main()

