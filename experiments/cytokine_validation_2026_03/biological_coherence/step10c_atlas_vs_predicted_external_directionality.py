#!/usr/bin/env python3
"""
Atlas DE vs predicted external dataset DE direction (same layout as step10b for real).

Reads:
  - biological_coherence/atlas_de_results.csv
  - biological_coherence/de_directionality_YYYY-MM-DD/de_tables/de_pred_all.csv

Writes:
  - .../directionality_metrics/atlas_vs_predicted_external_dataset_directionality_all_conditions.csv
  - .../directionality_metrics/atlas_vs_predicted_external_dataset_sign_match.png
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


def _effect_with_fallback(df: pd.DataFrame, lfc_col: str, score_col: str) -> pd.Series:
    """Use logFC when finite; fallback to score sign when logFC is NaN."""
    lfc = pd.to_numeric(df[lfc_col], errors="coerce")
    score = pd.to_numeric(df[score_col], errors="coerce")
    out = lfc.copy()
    miss = ~np.isfinite(out)
    out.loc[miss] = score.loc[miss]
    return out


def _metrics(merged: pd.DataFrame) -> dict[str, float]:
    s_pred = np.sign(merged["effect_pred"].values)
    s_atlas = np.sign(merged["effect_atlas"].values)

    all_match = float((s_pred == s_atlas).mean()) if len(merged) else np.nan

    nz = (s_pred != 0) & (s_atlas != 0)
    nz_match = float((s_pred[nz] == s_atlas[nz]).mean()) if nz.sum() else np.nan

    de_pred = merged["fdr_pred"].values < 0.05
    de_atlas = merged["fdr_atlas"].values < 0.05
    de_union = de_pred | de_atlas
    de_inter = de_pred & de_atlas

    def _rate(mask):
        if mask.sum() == 0:
            return np.nan
        return float((s_pred[mask] == s_atlas[mask]).mean())

    return {
        "n_shared_genes": int(len(merged)),
        "all_genes_sign_match": all_match,
        "nonzero_genes_sign_match": nz_match,
        "n_de_pred_fdr05": int(de_pred.sum()),
        "n_de_atlas_fdr05": int(de_atlas.sum()),
        "n_de_union_fdr05": int(de_union.sum()),
        "n_de_intersection_fdr05": int(de_inter.sum()),
        "de_pred_sign_match": _rate(de_pred),
        "de_union_sign_match": _rate(de_union),
        "de_intersection_sign_match": _rate(de_inter),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-tag", default=dt.date.today().isoformat(), help="Date tag for de_directionality_<date>")
    args = ap.parse_args()

    root = BASE / f"de_directionality_{args.date_tag}"
    de_pred_path = root / "de_tables" / "de_pred_all.csv"
    outdir = root / "directionality_metrics"
    outdir.mkdir(parents=True, exist_ok=True)

    atlas = pd.read_csv(ATLAS_DE)
    pred = pd.read_csv(de_pred_path)

    atlas["effect_atlas"] = _effect_with_fallback(atlas, "logFC_atlas", "score_atlas")
    pred["effect_pred"] = _effect_with_fallback(pred, "logFC_pred", "score_pred")

    rows = []
    for (ds, ct, cyto), sub_pred in pred.groupby(["dataset", "cell_type", "cytokine"]):
        sub_atlas = atlas[(atlas["cell_type"] == ct) & (atlas["cytokine"] == cyto)]
        if sub_atlas.empty:
            continue

        merged = (
            sub_pred[["gene", "effect_pred", "fdr_pred"]]
            .merge(sub_atlas[["gene", "effect_atlas", "fdr_atlas"]], on="gene", how="inner")
            .dropna(subset=["effect_pred", "effect_atlas"])
        )
        if len(merged) < 20:
            continue

        row = {"dataset": ds, "cell_type": ct, "cytokine": cyto}
        row.update(_metrics(merged))
        rows.append(row)

    out_csv = outdir / "atlas_vs_predicted_external_dataset_directionality_all_conditions.csv"
    if rows:
        mdf = pd.DataFrame(rows).sort_values(["dataset", "cell_type"])
        mdf.to_csv(out_csv, index=False)

        labels = [f"{r.cell_type}\n({r.dataset})" for r in mdf.itertuples()]
        y = np.arange(len(mdf))
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(mdf) * 0.35)))
        axes[0].barh(y, mdf["all_genes_sign_match"].values * 100, color="#31a354")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].set_xlim(0, 100)
        axes[0].set_xlabel("Sign match %")
        axes[0].set_title("Atlas vs predicted external dataset: all genes")

        axes[1].barh(y, np.nan_to_num(mdf["de_pred_sign_match"].values) * 100, color="#a1d99b")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Sign match %")
        axes[1].set_title("Atlas vs predicted external dataset: DE-only (predicted external FDR<0.05)")
        plt.tight_layout()
        fig.savefig(outdir / "atlas_vs_predicted_external_dataset_sign_match.png", dpi=220)
        plt.close(fig)
        print(f"[DONE] wrote {out_csv} and atlas_vs_predicted_external_dataset_sign_match.png")
    else:
        pd.DataFrame(
            columns=[
                "dataset", "cell_type", "cytokine", "n_shared_genes", "all_genes_sign_match",
                "nonzero_genes_sign_match", "n_de_pred_fdr05", "n_de_atlas_fdr05",
                "n_de_union_fdr05", "n_de_intersection_fdr05", "de_pred_sign_match",
                "de_union_sign_match", "de_intersection_sign_match",
            ]
        ).to_csv(out_csv, index=False)
        print("[WARN] no comparable atlas-vs-predicted-external conditions found.")


if __name__ == "__main__":
    main()
