#!/usr/bin/env python3
"""
Create clearly labeled binary directionality reports with confusion metrics and plots.

Comparisons:
1) predicted external DGE vs real external DGE
2) atlas DGE vs predicted external DGE
3) atlas DGE vs real external DGE
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026_03/biological_coherence")
ATLAS_DE = BASE / "atlas_de_results.csv"


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def _compute_binary_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> dict[str, float]:
    y_true = (df[true_col] > 0).astype(int)
    y_pred = (df[pred_col] > 0).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = tp + tn + fp + fn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else np.nan

    return {
        "n_binary_genes": int(n),
        "tp_up_up": tp,
        "tn_down_down": tn,
        "fp_down_to_up": fp,
        "fn_up_to_down": fn,
        "accuracy": _safe_div(tp + tn, n),
        "precision_up": precision,
        "recall_up": recall,
        "f1_up": f1,
        "specificity_down": specificity,
        "balanced_accuracy": _safe_div(recall + specificity, 2.0)
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan,
        "chance_baseline_binary": 0.5,
    }


def _effect_with_fallback(df: pd.DataFrame, lfc_col: str, score_col: str) -> pd.Series:
    lfc = pd.to_numeric(df[lfc_col], errors="coerce")
    score = pd.to_numeric(df[score_col], errors="coerce")
    out = lfc.copy()
    miss = ~np.isfinite(out)
    out.loc[miss] = score.loc[miss]
    return out


def _plot_metrics(cond_df: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    labels = [f"{r.cell_type}\n({r.dataset})" for r in cond_df.itertuples()]
    y = np.arange(len(cond_df))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(cond_df) * 0.35)))
    axes[0].barh(y, cond_df["accuracy"].values * 100, color="#3182bd")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].set_title(f"{title_prefix}: accuracy")

    axes[1].barh(y, np.nan_to_num(cond_df["f1_up"].values) * 100, color="#31a354")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("F1 (up class) (%)")
    axes[1].set_title(f"{title_prefix}: F1 (up class)")
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_accuracy_f1_by_condition.png", dpi=220)
    plt.close(fig)

    # stacked confusion proportions per condition
    conf = cond_df.copy()
    denom = conf["n_binary_genes"].replace(0, np.nan)
    conf["tp_rate"] = conf["tp_up_up"] / denom
    conf["tn_rate"] = conf["tn_down_down"] / denom
    conf["fp_rate"] = conf["fp_down_to_up"] / denom
    conf["fn_rate"] = conf["fn_up_to_down"] / denom

    fig, ax = plt.subplots(figsize=(11, max(5, len(conf) * 0.35)))
    left = np.zeros(len(conf))
    for col, color, lab in [
        ("tp_rate", "#2ca25f", "TP (up->up)"),
        ("tn_rate", "#3182bd", "TN (down->down)"),
        ("fp_rate", "#fd8d3c", "FP (down->up)"),
        ("fn_rate", "#de2d26", "FN (up->down)"),
    ]:
        vals = np.nan_to_num(conf[col].values)
        ax.barh(y, vals * 100, left=left * 100, color=color, label=lab)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Per-condition proportion of genes (%)")
    ax.set_title(f"{title_prefix}: confusion composition")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_composition_by_condition.png", dpi=220)
    plt.close(fig)


def _write_report(cond_df: pd.DataFrame, out_dir: Path, comparison_label: str) -> None:
    cond_csv = out_dir / "binary_metrics_all_conditions.csv"
    ct_csv = out_dir / "binary_metrics_by_cell_type.csv"
    summary_txt = out_dir / "README_metrics.txt"

    cond_df = cond_df.sort_values(["dataset", "cell_type"])
    cond_df.to_csv(cond_csv, index=False)

    agg = (
        cond_df.groupby("cell_type", as_index=False)[
            ["n_binary_genes", "tp_up_up", "tn_down_down", "fp_down_to_up", "fn_up_to_down"]
        ]
        .sum()
    )
    # recompute from summed confusion counts manually
    out = []
    for _, r in agg.iterrows():
        tp = int(r["tp_up_up"])
        tn = int(r["tn_down_down"])
        fp = int(r["fp_down_to_up"])
        fn = int(r["fn_up_to_down"])
        n = tp + tn + fp + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        out.append(
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
            }
        )
    ct = pd.concat([agg, pd.DataFrame(out)], axis=1)
    ct.to_csv(ct_csv, index=False)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Comparison: {comparison_label}\n")
        f.write("Binary labels: upregulated (logFC>0) vs downregulated (logFC<0)\n")
        f.write("Zeros removed from either side before metric computation.\n")
        f.write("Columns TP/TN/FP/FN are per-condition gene counts.\n")
        f.write("Chance baseline for balanced binary direction is 0.5.\n")
        f.write(f"\nConditions: {len(cond_df)}\n")
        f.write(f"Mean accuracy: {cond_df['accuracy'].mean():.4f}\n")
        f.write(f"Mean precision_up: {cond_df['precision_up'].mean():.4f}\n")
        f.write(f"Mean f1_up: {cond_df['f1_up'].mean():.4f}\n")


def _build_pred_vs_real(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "de_tables").glob("de_merged_*.csv")):
        stem = p.stem.removeprefix("de_merged_")
        ds = stem.split("_")[0]
        cyto = "IFN-gamma" if stem.endswith("IFN-gamma") else "IFN-beta"
        ct_raw = stem[len(ds) + 1 : -(len(cyto) + 1)]
        ct = ct_raw.replace("_", " ").replace("-", "/")

        df = pd.read_csv(p)
        for c in ["logFC_real", "logFC_pred"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["logFC_real", "logFC_pred"])
        df = df[(df["logFC_real"] != 0) & (df["logFC_pred"] != 0)]
        if len(df) == 0:
            continue
        rec = {"dataset": ds, "cell_type": ct, "cytokine": cyto}
        rec.update(_compute_binary_metrics(df, "logFC_real", "logFC_pred"))
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_atlas_vs_pred(root: Path) -> pd.DataFrame:
    atlas = pd.read_csv(ATLAS_DE)
    pred = pd.read_csv(root / "de_tables" / "de_pred_all.csv")
    atlas["effect_atlas"] = _effect_with_fallback(atlas, "logFC_atlas", "score_atlas")
    pred["effect_pred"] = _effect_with_fallback(pred, "logFC_pred", "score_pred")

    rows = []
    for (ds, ct, cyto), sub in pred.groupby(["dataset", "cell_type", "cytokine"]):
        a = atlas[(atlas["cell_type"] == ct) & (atlas["cytokine"] == cyto)]
        if a.empty:
            continue
        m = sub[["gene", "effect_pred"]].merge(a[["gene", "effect_atlas"]], on="gene", how="inner")
        m = m.dropna(subset=["effect_pred", "effect_atlas"])
        m = m[(m["effect_pred"] != 0) & (m["effect_atlas"] != 0)]
        if len(m) == 0:
            continue
        rec = {"dataset": ds, "cell_type": ct, "cytokine": cyto}
        rec.update(_compute_binary_metrics(m, "effect_atlas", "effect_pred"))
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_atlas_vs_real(root: Path) -> pd.DataFrame:
    atlas = pd.read_csv(ATLAS_DE)
    real = pd.read_csv(root / "de_tables" / "de_real_all.csv")
    atlas["effect_atlas"] = _effect_with_fallback(atlas, "logFC_atlas", "score_atlas")
    real["effect_real"] = _effect_with_fallback(real, "logFC_real", "score_real")

    rows = []
    for (ds, ct, cyto), sub in real.groupby(["dataset", "cell_type", "cytokine"]):
        a = atlas[(atlas["cell_type"] == ct) & (atlas["cytokine"] == cyto)]
        if a.empty:
            continue
        m = sub[["gene", "effect_real"]].merge(a[["gene", "effect_atlas"]], on="gene", how="inner")
        m = m.dropna(subset=["effect_real", "effect_atlas"])
        m = m[(m["effect_real"] != 0) & (m["effect_atlas"] != 0)]
        if len(m) == 0:
            continue
        rec = {"dataset": ds, "cell_type": ct, "cytokine": cyto}
        rec.update(_compute_binary_metrics(m, "effect_atlas", "effect_real"))
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-tag", default=dt.date.today().isoformat())
    args = ap.parse_args()

    root = BASE / f"de_directionality_{args.date_tag}"
    metrics_root = root / "directionality_metrics"

    out_pred_real = metrics_root / "binary_direction_predicted_dge_vs_real_dge"
    out_atlas_pred = metrics_root / "binary_direction_atlas_dge_vs_predicted_external_dge"
    out_atlas_real = metrics_root / "binary_direction_atlas_dge_vs_real_external_dge"
    out_pred_real.mkdir(parents=True, exist_ok=True)
    out_atlas_pred.mkdir(parents=True, exist_ok=True)
    out_atlas_real.mkdir(parents=True, exist_ok=True)

    pred_real = _build_pred_vs_real(root)
    if len(pred_real):
        _write_report(pred_real, out_pred_real, "predicted external DGE vs real external DGE")
        _plot_metrics(pred_real, out_pred_real, "Predicted DGE vs Real DGE")

    atlas_pred = _build_atlas_vs_pred(root)
    if len(atlas_pred):
        _write_report(atlas_pred, out_atlas_pred, "atlas DGE vs predicted external DGE")
        _plot_metrics(atlas_pred, out_atlas_pred, "Atlas DGE vs Predicted External DGE")

    atlas_real = _build_atlas_vs_real(root)
    if len(atlas_real):
        _write_report(atlas_real, out_atlas_real, "atlas DGE vs real external DGE")
        _plot_metrics(atlas_real, out_atlas_real, "Atlas DGE vs Real External DGE")

    print(f"[DONE] wrote: {out_pred_real}")
    print(f"[DONE] wrote: {out_atlas_pred}")
    print(f"[DONE] wrote: {out_atlas_real}")


if __name__ == "__main__":
    main()
