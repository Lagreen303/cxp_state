#!/usr/bin/env python3
"""
Pathway-level biological coherence for Kang CxP STATE infer posthoc.

Mirrors cytokine_validation biological_coherence:
  - step10_directionality_refocus (pred vs real DE sign metrics)
  - step10e pred vs real binary directionality
  - step6-style heatmap (pred vs real pathway coherence metrics)

Inputs: pathway_celleval/*_{real,pred}_de.csv and *_results.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import pearsonr, spearmanr

FDR_THRESH = 0.05
DATASET = "Kang_CxP"
PERT = "IFN-beta"


def _safe_log2fc(fc: pd.Series) -> pd.Series:
    x = pd.to_numeric(fc, errors="coerce").copy()
    x[x <= 0] = np.nan
    return np.log2(x)


def direction_metrics(merged: pd.DataFrame) -> dict[str, float]:
    s_real = np.sign(merged["logFC_real"].values)
    s_pred = np.sign(merged["logFC_pred"].values)
    all_sign_match = float((s_real == s_pred).mean())
    nz = (s_real != 0) & (s_pred != 0)
    nz_sign_match = float((s_real[nz] == s_pred[nz]).mean()) if nz.sum() > 0 else np.nan
    de_real = merged["fdr_real"].values < FDR_THRESH
    de_pred = merged["fdr_pred"].values < FDR_THRESH
    de_union = de_real | de_pred
    de_inter = de_real & de_pred

    def _sign_rate(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return np.nan
        return float((s_real[mask] == s_pred[mask]).mean())

    return {
        "n_shared_pathways": int(len(merged)),
        "all_pathways_sign_match": all_sign_match,
        "nonzero_pathways_sign_match": nz_sign_match,
        "n_de_real_fdr05": int(de_real.sum()),
        "n_de_pred_fdr05": int(de_pred.sum()),
        "n_de_union_fdr05": int(de_union.sum()),
        "n_de_intersection_fdr05": int(de_inter.sum()),
        "de_real_sign_match": _sign_rate(de_real),
        "de_union_sign_match": _sign_rate(de_union),
        "de_intersection_sign_match": _sign_rate(de_inter),
    }


def _compute_binary_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> dict[str, float]:
    y_true = (df[true_col] > 0).astype(int)
    y_pred = (df[pred_col] > 0).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = tp + tn + fp + fn

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den else np.nan

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "n_binary_pathways": int(n),
        "tp_up_up": tp,
        "tn_down_down": tn,
        "fp_down_to_up": fp,
        "fn_up_to_down": fn,
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
    }


@dataclass
class PathwayCoherence:
    dataset: str
    cell_type: str
    cytokine: str
    spearman_logFC: float = np.nan
    pearson_logFC: float = np.nan
    jaccard_de: float = np.nan
    precision_de: float = np.nan
    recall_de: float = np.nan
    n_de_real: int = 0
    n_de_pred: int = 0
    n_shared_pathways: int = 0


def load_de_pair(celleval_dir: Path, cell_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_p = celleval_dir / f"{cell_type}_real_de.csv"
    pred_p = celleval_dir / f"{cell_type}_pred_de.csv"
    if not real_p.is_file() or not pred_p.is_file():
        raise FileNotFoundError(f"missing DE pair for {cell_type}")
    real = pd.read_csv(real_p)
    pred = pd.read_csv(pred_p)
    real = real[real["target"].astype(str) == PERT].copy()
    pred = pred[pred["target"].astype(str) == PERT].copy()
    return real, pred


def merged_de_tables(real: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    r = real.rename(
        columns={
            "feature": "pathway_id",
            "fold_change": "fc_real",
            "fdr": "fdr_real",
        }
    )
    p = pred.rename(
        columns={
            "feature": "pathway_id",
            "fold_change": "fc_pred",
            "fdr": "fdr_pred",
        }
    )
    m = r[["pathway_id", "fc_real", "fdr_real"]].merge(
        p[["pathway_id", "fc_pred", "fdr_pred"]], on="pathway_id", how="inner"
    )
    m["logFC_real"] = _safe_log2fc(m["fc_real"])
    m["logFC_pred"] = _safe_log2fc(m["fc_pred"])
    return m.dropna(subset=["logFC_real", "logFC_pred"])


def coherence_from_merged(merged: pd.DataFrame, cell_type: str) -> PathwayCoherence:
    out = PathwayCoherence(
        dataset=DATASET, cell_type=cell_type, cytokine=PERT, n_shared_pathways=len(merged)
    )
    try:
        out.spearman_logFC = float(spearmanr(merged["logFC_real"], merged["logFC_pred"]).correlation)
        out.pearson_logFC = float(pearsonr(merged["logFC_real"], merged["logFC_pred"]).statistic)
    except Exception:
        pass
    real_sig = set(merged.loc[merged["fdr_real"] < FDR_THRESH, "pathway_id"])
    pred_sig = set(merged.loc[merged["fdr_pred"] < FDR_THRESH, "pathway_id"])
    out.n_de_real = len(real_sig)
    out.n_de_pred = len(pred_sig)
    union = real_sig | pred_sig
    inter = real_sig & pred_sig
    out.jaccard_de = len(inter) / len(union) if union else np.nan
    out.precision_de = len(inter) / len(pred_sig) if pred_sig else np.nan
    out.recall_de = len(inter) / len(real_sig) if real_sig else np.nan
    return out


def make_heatmap(coherence_df: pd.DataFrame, outpath: Path) -> None:
    metrics = [
        ("spearman_logFC", "Spearman logFC", (-1, 1), "RdYlGn"),
        ("jaccard_de", "Jaccard DE overlap", (0, 1), "YlOrRd"),
        ("precision_de", "Precision DE", (0, 1), "YlOrRd"),
        ("recall_de", "Recall DE", (0, 1), "YlOrRd"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 2.8))
    sub = coherence_df.sort_values("cell_type")
    for ax, (col, label, (vmin, vmax), cmap) in zip(axes, metrics):
        pivot = sub.set_index("cell_type")[[col]].T
        sns.heatmap(
            pivot,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            cbar=col == metrics[-1][0],
            yticklabels=["Kang CxP (IFN-β)"],
            xticklabels=list(pivot.columns),
        )
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    fig.suptitle("Pathway-level coherence (decoded pred vs real)", fontsize=12, fontweight="bold", y=1.08)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--celleval-dir",
        type=Path,
        default=Path(
            "/iridisfs/ddnb/Luke/cellxpathway/outputs/kang_state/v2_collectri/posthoc/pathway_celleval"
        ),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: .../kang_state/v2_collectri/biological_coherence/de_directionality_<date>",
    )
    args = ap.parse_args()

    celleval_dir = args.celleval_dir.resolve()
    date_tag = dt.date.today().isoformat()
    if args.out_root:
        out = args.out_root.resolve()
    else:
        out = (
            Path("/iridisfs/ddnb/Luke/cellxpathway/outputs/kang_state/v2_collectri/biological_coherence")
            / f"de_directionality_{date_tag}"
        )
    de_dir = out / "de_tables"
    met_dir = out / "directionality_metrics"
    plot_dir = out.parent / "plots" / "heatmaps"
    de_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cell_types = sorted(
        {f.name.replace("_real_de.csv", "") for f in celleval_dir.glob("*_real_de.csv")}
    )

    metric_rows = []
    coherence_rows = []
    binary_rows = []
    all_real = []
    all_pred = []
    skipped = []

    for ct in cell_types:
        try:
            real, pred = load_de_pair(celleval_dir, ct)
        except FileNotFoundError:
            skipped.append({"cell_type": ct, "reason": "missing_de_csv"})
            continue
        merged = merged_de_tables(real, pred)
        if merged.empty:
            skipped.append({"cell_type": ct, "reason": "empty_merge"})
            continue

        row = {"dataset": DATASET, "cell_type": ct, "cytokine": PERT}
        row.update(direction_metrics(merged))
        metric_rows.append(row)

        coh = coherence_from_merged(merged, ct)
        coherence_rows.append(vars(coh))

        bdf = merged.dropna(subset=["logFC_real", "logFC_pred"])
        bdf = bdf[(bdf["logFC_real"] != 0) & (bdf["logFC_pred"] != 0)]
        if not bdf.empty:
            rec = {"dataset": DATASET, "cell_type": ct, "cytokine": PERT}
            rec.update(_compute_binary_metrics(bdf, "logFC_real", "logFC_pred"))
            binary_rows.append(rec)

        ct_safe = ct.replace(" ", "_").replace("/", "-")
        base = f"{DATASET}_{ct_safe}_{PERT.replace(' ', '_')}"
        real_out = real.rename(columns={"feature": "pathway_id", "fold_change": "fc", "fdr": "fdr"})
        pred_out = pred.rename(columns={"feature": "pathway_id", "fold_change": "fc", "fdr": "fdr"})
        real_out["cell_type"] = ct
        pred_out["cell_type"] = ct
        all_real.append(real_out)
        all_pred.append(pred_out)
        merged.assign(cell_type=ct, cytokine=PERT).to_csv(de_dir / f"de_merged_{base}.csv", index=False)
        real_out.to_csv(de_dir / f"de_real_{base}.csv", index=False)
        pred_out.to_csv(de_dir / f"de_pred_{base}.csv", index=False)

    if all_real:
        pd.concat(all_real, ignore_index=True).to_csv(de_dir / "de_real_all.csv", index=False)
    if all_pred:
        pd.concat(all_pred, ignore_index=True).to_csv(de_dir / "de_pred_all.csv", index=False)

    mdf = pd.DataFrame(metric_rows).sort_values("cell_type")
    mdf.to_csv(met_dir / "directionality_metrics_all_conditions.csv", index=False)

    cdf = pd.DataFrame(coherence_rows).sort_values("cell_type")
    cdf.to_csv(out.parent / "pathway_coherence_metrics.csv", index=False)

    if binary_rows:
        bdf_all = pd.DataFrame(binary_rows).sort_values("cell_type")
        bin_dir = met_dir / "binary_direction_predicted_dge_vs_real_dge"
        bin_dir.mkdir(parents=True, exist_ok=True)
        bdf_all.to_csv(bin_dir / "binary_metrics_all_conditions.csv", index=False)
        bdf_all.to_csv(bin_dir / "binary_metrics_by_cell_type.csv", index=False)
        with open(bin_dir / "README_metrics.txt", "w", encoding="utf-8") as f:
            f.write("Kang CxP pathway DE: predicted vs real (IFN-beta vs control)\n")
            f.write(f"Mean accuracy: {bdf_all['accuracy'].mean():.4f}\n")

    pd.DataFrame(skipped).to_csv(met_dir / "skipped_conditions.csv", index=False)

    if not mdf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(mdf) * 0.35)))
        labels = [f"{r.cell_type}" for r in mdf.itertuples()]
        y = np.arange(len(mdf))
        axes[0].barh(y, mdf["all_pathways_sign_match"].values * 100, color="#2b8cbe")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].set_xlim(0, 100)
        axes[0].set_title("All pathways (primary)")
        axes[1].barh(y, np.nan_to_num(mdf["de_real_sign_match"].values) * 100, color="#de2d26")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlim(0, 100)
        axes[1].set_title("DE-only (real FDR<0.05)")
        plt.tight_layout()
        fig.savefig(met_dir / "directionality_sign_match_primary_vs_de.png", dpi=220)
        plt.close(fig)

    if not cdf.empty:
        make_heatmap(cdf, plot_dir / "heatmap_Kang_CxP_all_metrics.png")

    # Celleval summary metrics heatmap (pr_auc, pearson_delta, de_direction_match)
    ce_rows = []
    for ct in cell_types:
        res_p = celleval_dir / f"{ct}_results.csv"
        if not res_p.is_file():
            continue
        df = pd.read_csv(res_p)
        sub = df[df["perturbation"].astype(str) == PERT]
        if sub.empty:
            continue
        r = sub.iloc[0]
        ce_rows.append(
            {
                "cell_type": ct,
                "pr_auc": r.get("pr_auc", np.nan),
                "pearson_delta": r.get("pearson_delta", np.nan),
                "de_direction_match": r.get("de_direction_match", np.nan),
                "roc_auc": r.get("roc_auc", np.nan),
            }
        )
    if ce_rows:
        ce_df = pd.DataFrame(ce_rows).sort_values("cell_type")
        ce_df.to_csv(met_dir / "celleval_per_cell_type.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(14, 2.8))
        for ax, col, title in zip(
            axes,
            ["pr_auc", "pearson_delta", "de_direction_match"],
            ["PR-AUC", "Pearson Δ", "DE direction match"],
        ):
            pivot = ce_df.set_index("cell_type")[[col]].T
            sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_title(title)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        fig.suptitle("Kang CxP pathway cell-eval (IFN-β)", y=1.05)
        plt.tight_layout()
        fig.savefig(plot_dir / "heatmap_Kang_CxP_celleval_metrics.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    manifest = {
        "date": date_tag,
        "phase": "kang_cxp_pathway_directionality",
        "dataset": DATASET,
        "perturbation": PERT,
        "celleval_dir": str(celleval_dir),
        "outputs": {"root": str(out), "de_tables": str(de_dir), "directionality_metrics": str(met_dir)},
        "notes": {
            "strategy": "pathway_decoded_pred_vs_real_from_cxp_posthoc_celleval",
            "de_source": "cell_eval MetricsEvaluator on 666 CollecTRI pathways",
        },
    }
    with open(out / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"[DONE] {out}")
    print(f"  heatmaps: {plot_dir}")


if __name__ == "__main__":
    main()
