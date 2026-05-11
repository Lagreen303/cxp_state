#!/usr/bin/env python3
"""
Compute binary direction-classification metrics from merged DE tables.

Positive class: upregulated (logFC > 0)
Negative class: downregulated (logFC < 0)

Zero logFC rows are excluded to keep a strict binary setting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026_03/biological_coherence")
MERGED_RE = re.compile(r"^de_merged_(GSE\d+)_(.+)_(IFN-beta|IFN-gamma)$")


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return np.nan
    return float(num / den)


def _confusion_and_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = (df["logFC_real"] > 0).astype(int)
    y_pred = (df["logFC_pred"] > 0).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, n)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else np.nan
    specificity = _safe_div(tn, tn + fp)
    balanced_accuracy = (
        _safe_div(recall + specificity, 2.0)
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan
    )

    return {
        "n_binary_genes": n,
        "n_up_real": int((y_true == 1).sum()),
        "n_down_real": int((y_true == 0).sum()),
        "tp_up_up": tp,
        "tn_down_down": tn,
        "fp_down_to_up": fp,
        "fn_up_to_down": fn,
        "accuracy": accuracy,
        "precision_up": precision,
        "recall_up": recall,
        "f1_up": f1,
        "specificity_down": specificity,
        "balanced_accuracy": balanced_accuracy,
        "chance_baseline_binary": 0.5,
    }


def _load_and_filter(path: Path, subset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"gene", "logFC_real", "logFC_pred", "fdr_real", "fdr_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    out = df.copy()
    out["logFC_real"] = pd.to_numeric(out["logFC_real"], errors="coerce")
    out["logFC_pred"] = pd.to_numeric(out["logFC_pred"], errors="coerce")
    out["fdr_real"] = pd.to_numeric(out["fdr_real"], errors="coerce")
    out["fdr_pred"] = pd.to_numeric(out["fdr_pred"], errors="coerce")

    out = out.dropna(subset=["logFC_real", "logFC_pred"])
    # strict binary setup: remove ties at exactly 0 on either side
    out = out[(out["logFC_real"] != 0) & (out["logFC_pred"] != 0)]

    if subset == "de_real":
        out = out[out["fdr_real"] < 0.05]
    elif subset == "de_union":
        out = out[(out["fdr_real"] < 0.05) | (out["fdr_pred"] < 0.05)]
    elif subset == "de_intersection":
        out = out[(out["fdr_real"] < 0.05) & (out["fdr_pred"] < 0.05)]

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-tag", default=dt.date.today().isoformat(), help="Date tag for de_directionality_<date>")
    ap.add_argument(
        "--subset",
        default="all",
        choices=["all", "de_real", "de_union", "de_intersection"],
        help="Gene subset used before computing binary metrics.",
    )
    args = ap.parse_args()

    root = BASE / f"de_directionality_{args.date_tag}"
    de_dir = root / "de_tables"
    out_dir = root / "directionality_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_paths = sorted(de_dir.glob("de_merged_*.csv"))
    if not merged_paths:
        raise FileNotFoundError(f"No de_merged_*.csv files found in {de_dir}")

    rows = []
    for p in merged_paths:
        m = MERGED_RE.match(p.stem)
        if not m:
            continue
        ds = m.group(1)
        ct_raw = m.group(2)
        cyto = m.group(3)
        ct = ct_raw.replace("-", "/").replace("_", " ")

        df = _load_and_filter(p, args.subset)
        if len(df) == 0:
            continue
        rec = {"dataset": ds, "cell_type": ct, "cytokine": cyto, "subset": args.subset}
        rec.update(_confusion_and_metrics(df))
        rows.append(rec)

    out_csv = out_dir / f"direction_binary_metrics_{args.subset}_all_conditions.csv"
    out_celltype_csv = out_dir / f"direction_binary_metrics_{args.subset}_by_cell_type.csv"

    mdf = pd.DataFrame(rows).sort_values(["dataset", "cell_type"]) if rows else pd.DataFrame()
    mdf.to_csv(out_csv, index=False)

    if len(mdf):
        agg = (
            mdf.groupby("cell_type", as_index=False)[
                [
                    "n_binary_genes",
                    "tp_up_up",
                    "tn_down_down",
                    "fp_down_to_up",
                    "fn_up_to_down",
                ]
            ]
            .sum()
        )
        # recompute metrics from summed confusion counts
        metrics = []
        for _, r in agg.iterrows():
            tp = int(r["tp_up_up"])
            tn = int(r["tn_down_down"])
            fp = int(r["fp_down_to_up"])
            fn = int(r["fn_up_to_down"])
            n = tp + tn + fp + fn
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            specificity = _safe_div(tn, tn + fp)
            metrics.append(
                {
                    "accuracy": _safe_div(tp + tn, n),
                    "precision_up": precision,
                    "recall_up": recall,
                    "f1_up": _safe_div(2 * precision * recall, precision + recall)
                    if np.isfinite(precision) and np.isfinite(recall)
                    else np.nan,
                    "specificity_down": specificity,
                    "balanced_accuracy": _safe_div(recall + specificity, 2.0)
                    if np.isfinite(recall) and np.isfinite(specificity)
                    else np.nan,
                    "chance_baseline_binary": 0.5,
                    "subset": args.subset,
                }
            )
        out_ct = pd.concat([agg, pd.DataFrame(metrics)], axis=1)
        out_ct.to_csv(out_celltype_csv, index=False)
    else:
        pd.DataFrame().to_csv(out_celltype_csv, index=False)

    print(f"[DONE] wrote {out_csv}")
    print(f"[DONE] wrote {out_celltype_csv}")


if __name__ == "__main__":
    main()
