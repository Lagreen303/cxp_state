#!/usr/bin/env python3
"""
Three-panel horizontal bar chart: DE-only sign match per condition (same logic as
directionality_sign_match_primary_vs_de.png, right panel).

Panels (left to right):
  1) Atlas vs real external: sign(atlas) == sign(real) on genes with real FDR < 0.05
     (de_real_sign_match from atlas_vs_real_external_dataset_directionality_*.csv)
  2) Predicted vs real external: same as original right panel — real FDR < 0.05
     (de_real_sign_match from directionality_metrics_all_conditions.csv)
  3) Atlas vs predicted external: predicted FDR < 0.05 (parallel to step10c DE-only)
     (de_pred_sign_match from atlas_vs_predicted_external_dataset_directionality_*.csv)

Drops conditions with NaN / no DE genes on the relevant FDR mask for any panel
(inner join on finite metrics).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _label_row(r: pd.Series) -> str:
    return f"{r['cell_type']}\n({r['dataset']}, {r['cytokine']})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing directionality_metrics_all_conditions.csv etc.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: metrics-dir/dge_sign_match_three_panel.png)",
    )
    args = ap.parse_args()
    root: Path = args.metrics_dir
    out_path = args.out or (root / "dge_sign_match_three_panel.png")

    keys = ["dataset", "cell_type", "cytokine"]
    p_atlas_real = root / "atlas_vs_real_external_dataset_directionality_all_conditions.csv"
    p_pred_real = root / "directionality_metrics_all_conditions.csv"
    p_atlas_pred = root / "atlas_vs_predicted_external_dataset_directionality_all_conditions.csv"

    ar = pd.read_csv(p_atlas_real)
    pr = pd.read_csv(p_pred_real)
    ap_df = pd.read_csv(p_atlas_pred)

    need_ar = keys + ["de_real_sign_match", "n_de_real_fdr05"]
    need_pr = keys + ["de_real_sign_match", "n_de_real_fdr05"]
    need_ap = keys + ["de_pred_sign_match", "n_de_pred_fdr05"]
    for name, df, cols in [
        ("atlas_vs_real", ar, need_ar),
        ("pred_vs_real", pr, need_pr),
        ("atlas_vs_pred", ap_df, need_ap),
    ]:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise SystemExit(f"{name}: missing columns {miss}")

    ar = ar.copy()
    pr = pr.copy()
    ap_df = ap_df.copy()

    ar["de_real_sign_match"] = pd.to_numeric(ar["de_real_sign_match"], errors="coerce")
    pr["de_real_sign_match"] = pd.to_numeric(pr["de_real_sign_match"], errors="coerce")
    ap_df["de_pred_sign_match"] = pd.to_numeric(ap_df["de_pred_sign_match"], errors="coerce")
    ar["n_de_real_fdr05"] = pd.to_numeric(ar["n_de_real_fdr05"], errors="coerce").fillna(0).astype(int)
    pr["n_de_real_fdr05"] = pd.to_numeric(pr["n_de_real_fdr05"], errors="coerce").fillna(0).astype(int)
    ap_df["n_de_pred_fdr05"] = pd.to_numeric(ap_df["n_de_pred_fdr05"], errors="coerce").fillna(0).astype(int)

    m = ar[keys + ["de_real_sign_match", "n_de_real_fdr05"]].rename(
        columns={"de_real_sign_match": "pct_atlas_real", "n_de_real_fdr05": "n_de_real_ar"}
    )
    m = m.merge(
        pr[keys + ["de_real_sign_match", "n_de_real_fdr05"]].rename(
            columns={"de_real_sign_match": "pct_pred_real", "n_de_real_fdr05": "n_de_real_pr"}
        ),
        on=keys,
        how="inner",
    )
    m = m.merge(
        ap_df[keys + ["de_pred_sign_match", "n_de_pred_fdr05"]].rename(
            columns={"de_pred_sign_match": "pct_atlas_pred", "n_de_pred_fdr05": "n_de_pred_ap"}
        ),
        on=keys,
        how="inner",
    )

    finite = (
        np.isfinite(m["pct_atlas_real"].values)
        & np.isfinite(m["pct_pred_real"].values)
        & np.isfinite(m["pct_atlas_pred"].values)
    )
    m = m.loc[finite].copy()
    m = m[(m["n_de_real_ar"] > 0) & (m["n_de_real_pr"] > 0) & (m["n_de_pred_ap"] > 0)].copy()

    if len(m) == 0:
        raise SystemExit("No conditions left after inner join and finite DE-only metrics filter.")

    m["_sort"] = m["dataset"].astype(str) + m["cell_type"].astype(str) + m["cytokine"].astype(str)
    m = m.sort_values("_sort").drop(columns=["_sort"])

    labels = [_label_row(r) for _, r in m.iterrows()]
    y = np.arange(len(m))
    n = len(m)
    fig_h = max(6.0, n * 0.38)
    fig, axes = plt.subplots(1, 3, figsize=(16, fig_h), sharey=True)

    panels = [
        (
            axes[0],
            m["pct_atlas_real"].values * 100.0,
            "A: Atlas vs real (DE-only: real FDR < 0.05)",
            "#3182bd",
        ),
        (
            axes[1],
            m["pct_pred_real"].values * 100.0,
            "B: Predicted vs real (DE-only: real FDR < 0.05)",
            "#de2d26",
        ),
        (
            axes[2],
            m["pct_atlas_pred"].values * 100.0,
            "C: Atlas vs predicted (DE-only: predicted FDR < 0.05)",
            "#31a354",
        ),
    ]

    for ax, vals, title, color in panels:
        ax.barh(y, vals, color=color, height=0.75)
        ax.set_xlim(0, 100)
        ax.axvline(50, color="0.45", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlabel("Sign match (%)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()

    fig.suptitle(
        "DE-only direction agreement (same construction as primary-vs-DE figure, right column; "
        "panel C uses predicted FDR < 0.05, matching atlas-vs-predicted DE-only plots)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] wrote {out_path} ({len(m)} conditions)")


if __name__ == "__main__":
    main()
