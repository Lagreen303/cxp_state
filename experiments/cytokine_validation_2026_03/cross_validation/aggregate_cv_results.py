#!/usr/bin/env python3
"""
Aggregate per-perturbation, per-cell-type evaluation results across all CV folds.

Expects: for each run_2026_02_05_fold{K}, an eval directory (e.g. eval_last.ckpt/ or eval_final.ckpt/)
containing <CellType>_results.csv files (one row per perturbation, columns include
'perturbation' and metric columns).

Outputs (into the same cross_validation directory):
  - cytokine_cv_aggregated_results.csv
    long table with: fold, cell_type, perturbation, and all metrics

  - cytokine_cv_summary_by_perturbation.csv
    mean (+ std) of selected metrics per perturbation across folds and cell types
"""

from pathlib import Path

import pandas as pd


CV_DIR = Path(__file__).resolve().parent
RUN_PREFIX = "run_2026_02_05_fold"
EVAL_DIR_NAMES = ("eval_last.ckpt", "eval_final.ckpt")
RESULTS_SUFFIX = "_results.csv"
AGG_SUFFIX = "_agg_results.csv"


def find_eval_dir(run_dir: Path) -> Path | None:
    """Return the eval subdir under run_dir (prefer eval_last.ckpt if present)."""
    for name in EVAL_DIR_NAMES:
        d = run_dir / name
        if d.is_dir():
            return d
    # Fallback: any directory starting with eval_
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("eval_"):
            return d
    return None


def collect_fold_results(run_dir: Path, fold: int) -> pd.DataFrame | None:
    """
    Load all *_results.csv from this fold's eval dir into one DataFrame
    with added 'fold' and 'cell_type' columns.
    """
    eval_dir = find_eval_dir(run_dir)
    if eval_dir is None:
        return None

    result_files = [
        f
        for f in eval_dir.glob(f"*{RESULTS_SUFFIX}")
        if not f.name.endswith(AGG_SUFFIX)
    ]
    if not result_files:
        return None

    rows = []
    for path in sorted(result_files):
        cell_type = path.stem.replace("_results", "").replace("_agg", "")
        df = pd.read_csv(path)
        if "perturbation" not in df.columns:
            # skip non per-perturbation result files
            continue
        df = df.copy()
        df["fold"] = fold
        df["cell_type"] = cell_type
        rows.append(df)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def main():
    all_dfs = []

    for k in range(1, 10):
        run_dir = CV_DIR / f"{RUN_PREFIX}{k}"
        if not run_dir.is_dir():
            print(f"[SKIP] Fold {k}: run dir {run_dir} not found")
            continue

        df = collect_fold_results(run_dir, k)
        if df is None:
            print(f"[SKIP] Fold {k}: no usable *_results.csv files")
            continue

        all_dfs.append(df)
        print(f"[OK] Fold {k}: {len(df)} rows collected from {run_dir}")

    if not all_dfs:
        raise FileNotFoundError("No fold results found. Make sure prediction/eval has completed.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Put meta columns first
    meta = ["fold", "cell_type", "perturbation"]
    rest = [c for c in combined.columns if c not in meta]
    combined = combined[meta + rest]

    out_long = CV_DIR / "cytokine_cv_aggregated_results.csv"
    combined.to_csv(out_long, index=False)
    print(f"Saved long-form results: {out_long} ({len(combined)} rows)")

    # Summary: mean (and std) of key metrics per perturbation across folds and cell types
    key_metrics = [
        "pearson_delta", "mse", "mae", "mse_delta", "mae_delta",
        "pr_auc", "roc_auc",
        "de_spearman_sig", "de_direction_match", "de_sig_genes_recall",
    ]
    available = [m for m in key_metrics if m in combined.columns]

    if available:
        summary = combined.groupby("perturbation", as_index=False)[available].agg(["mean", "std"])
        summary = summary.round(6)
        # Flatten MultiIndex columns to e.g. pearson_delta_mean, pearson_delta_std
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        out_summary = CV_DIR / "cytokine_cv_summary_by_perturbation.csv"
        summary.to_csv(out_summary, index=False)
        print(f"Saved per-perturbation summary: {out_summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Aggregate per-perturbation, per-cell-type evaluation results across all CV folds.

Expects: for each run_2026_02_05_fold{K}, an eval directory (e.g. eval_last.ckpt/ or eval_final.ckpt/)
containing <CellType>_results.csv files (one row per perturbation, columns include
'perturbation' and metric columns).

Outputs (into the same cross_validation directory):
  - cytokine_cv_aggregated_results.csv
    long table with: fold, cell_type, perturbation, and all metrics

  - cytokine_cv_summary_by_perturbation.csv
    mean (+ std) of selected metrics per perturbation across folds and cell types
"""

from pathlib import Path

import pandas as pd


CV_DIR = Path(__file__).resolve().parent
RUN_PREFIX = "run_2026_02_05_fold"
EVAL_DIR_NAMES = ("eval_last.ckpt", "eval_final.ckpt")
RESULTS_SUFFIX = "_results.csv"
AGG_SUFFIX = "_agg_results.csv"


def find_eval_dir(run_dir: Path) -> Path | None:
    """Return the eval subdir under run_dir (prefer eval_last.ckpt if present)."""
    for name in EVAL_DIR_NAMES:
        d = run_dir / name
        if d.is_dir():
            return d
    # Fallback: any directory starting with eval_
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("eval_"):
            return d
    return None


def collect_fold_results(run_dir: Path, fold: int) -> pd.DataFrame | None:
    """
    Load all *_results.csv from this fold's eval dir into one DataFrame
    with added 'fold' and 'cell_type' columns.
    """
    eval_dir = find_eval_dir(run_dir)
    if eval_dir is None:
        return None

    result_files = [
        f
        for f in eval_dir.glob(f"*{RESULTS_SUFFIX}")
        if not f.name.endswith(AGG_SUFFIX)
    ]
    if not result_files:
        return None

    rows = []
    for path in sorted(result_files):
        cell_type = path.stem.replace("_results", "").replace("_agg", "")
        df = pd.read_csv(path)
        if "perturbation" not in df.columns:
            # skip non per-perturbation result files
            continue
        df = df.copy()
        df["fold"] = fold
        df["cell_type"] = cell_type
        rows.append(df)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def main():
    all_dfs = []

    for k in range(1, 10):
        run_dir = CV_DIR / f"{RUN_PREFIX}{k}"
        if not run_dir.is_dir():
            print(f"[SKIP] Fold {k}: run dir {run_dir} not found")
            continue

        df = collect_fold_results(run_dir, k)
        if df is None:
            print(f"[SKIP] Fold {k}: no usable *_results.csv files")
            continue

        all_dfs.append(df)
        print(f"[OK] Fold {k}: {len(df)} rows collected from {run_dir}")

    if not all_dfs:
        raise FileNotFoundError("No fold results found. Make sure prediction/eval has completed.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Put meta columns first
    meta = ["fold", "cell_type", "perturbation"]
    rest = [c for c in combined.columns if c not in meta]
    combined = combined[meta + rest]

    out_long = CV_DIR / "cytokine_cv_aggregated_results.csv"
    combined.to_csv(out_long, index=False)
    print(f"Saved long-form results: {out_long} ({len(combined)} rows)")

    # Summary: mean (and std) of key metrics per perturbation across folds and cell types
    key_metrics = [
        "pearson_delta", "mse", "mae", "mse_delta", "mae_delta",
        "pr_auc", "roc_auc",
        "de_spearman_sig", "de_direction_match", "de_sig_genes_recall",
    ]
    available = [m for m in key_metrics if m in combined.columns]

    if available:
        summary = combined.groupby("perturbation", as_index=False)[available].agg(["mean", "std"])
        summary = summary.round(6)
        # Flatten MultiIndex columns to e.g. pearson_delta_mean, pearson_delta_std
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        out_summary = CV_DIR / "cytokine_cv_summary_by_perturbation.csv"
        summary.to_csv(out_summary, index=False)
        print(f"Saved per-perturbation summary: {out_summary}")


if __name__ == "__main__":
    main()