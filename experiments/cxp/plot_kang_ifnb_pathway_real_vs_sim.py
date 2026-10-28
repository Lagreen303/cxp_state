#!/usr/bin/env python3
"""Real vs STATE-simulated pathway scores for Kang IFN-beta cells."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

STATE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STATE_ROOT / "experiments" / "cxp"))

from cxp_collectri_posthoc import (  # noqa: E402
    build_pathway_anndata_pair,
    load_posthoc_config,
)
from collectri_posthoc.pathway_metric_diagnostics import _pseudobulk_delta  # noqa: E402

CELLX_ROOT = Path("/iridisfs/ddnb/Luke/cellxpathway")
sys.path.insert(0, str(CELLX_ROOT))
from cellxpathway.eval import plots as cxp_plots  # noqa: E402

_SAFE = re.compile(r"[^\w\-]+")
PERT = "IFN-beta"


def _safe(s: str) -> str:
    return _SAFE.sub("_", s.strip()).strip("_")


def _pearson_vec(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _ifn_pseudobulk(
    x: np.ndarray,
    obs: pd.DataFrame,
    pert_col: str,
    cell_type_key: str,
    pert: str,
) -> dict[str, np.ndarray]:
    mask = obs[pert_col].astype(str) == pert
    if not mask.any():
        raise ValueError(f"no cells with {pert_col}=={pert}")
    out: dict[str, np.ndarray] = {}
    ct_vals = obs.loc[mask, cell_type_key].astype(str)
    x_ifn = x[mask.to_numpy()]
    for ct in sorted(ct_vals.unique()):
        m = ct_vals == ct
        out[ct] = x_ifn[m.to_numpy()].mean(axis=0)
    out["__all__"] = x_ifn.mean(axis=0)
    return out


def _scatter_panel(
    ax: plt.Axes,
    real: np.ndarray,
    sim: np.ndarray,
    title: str,
    *,
    metrics: dict[str, float],
) -> None:
    ax.scatter(
        real,
        sim,
        s=10,
        alpha=0.55,
        color=cxp_plots.CATEGORICAL_HUES[0],
        edgecolors="none",
    )
    lo = float(min(real.min(), sim.min()))
    hi = float(max(real.max(), sim.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="0.55", lw=0.8, zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("real pathway score")
    ax.set_ylabel("simulated pathway score")
    ax.set_title(
        f"{title}\n"
        f"r_profile={metrics['pearson_profile']:.3f}  "
        f"r_delta={metrics['pearson_delta']:.3f}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=STATE_ROOT / "experiments/cxp/configs/cxp_collectri_posthoc_kang.yaml",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pert", type=str, default=PERT)
    args = parser.parse_args()

    cfg = load_posthoc_config(args.config)
    eval_dir = Path(cfg["eval_dir"])
    out_dir = args.out_dir or (eval_dir / "figures_ifnb_pathway_real_vs_sim")
    out_dir.mkdir(parents=True, exist_ok=True)

    pert_col = str(cfg["pert_col"])
    cell_type_key = str(cfg["cell_type_key"])
    control = str(cfg["control_pert"])
    pert = str(args.pert)

    adata_pred, adata_real, meta = build_pathway_anndata_pair(cfg, eval_dir)
    obs = adata_real.obs
    real_x = np.asarray(adata_real.X, dtype=np.float64)
    sim_x = np.asarray(adata_pred.X, dtype=np.float64)

    real_pb = _ifn_pseudobulk(real_x, obs, pert_col, cell_type_key, pert)
    sim_pb = _ifn_pseudobulk(sim_x, obs, pert_col, cell_type_key, pert)

    real_d_by_ct: dict[str, dict[str, np.ndarray]] = {}
    sim_d_by_ct: dict[str, dict[str, np.ndarray]] = {}
    cell_types = sorted(c for c in real_pb if c != "__all__")
    for ct in cell_types:
        _, real_d_by_ct[ct] = _pseudobulk_delta(
            real_x, obs, pert_col, control, ct, cell_type_key
        )
        _, sim_d_by_ct[ct] = _pseudobulk_delta(
            sim_x, obs, pert_col, control, ct, cell_type_key
        )

    rows: list[dict[str, object]] = []
    for ct in cell_types + ["__all__"]:
        r_prof = _pearson_vec(real_pb[ct], sim_pb[ct])
        if ct == "__all__":
            # pooled IFN-beta pseudobulk; delta vs pooled control pseudobulk
            mask_ifn = obs[pert_col].astype(str) == pert
            mask_ctrl = obs[pert_col].astype(str) == control
            r_delta = _pearson_vec(
                real_x[mask_ifn.to_numpy()].mean(0) - real_x[mask_ctrl.to_numpy()].mean(0),
                sim_x[mask_ifn.to_numpy()].mean(0) - sim_x[mask_ctrl.to_numpy()].mean(0),
            )
            label = "all IFN-beta cells"
        else:
            r_delta = (
                _pearson_vec(real_d_by_ct[ct][pert], sim_d_by_ct[ct][pert])
                if pert in real_d_by_ct[ct]
                else float("nan")
            )
            label = ct
        rows.append(
            {
                "cell_type": label,
                "perturbation": pert,
                "n_ifn_cells": int(
                    ((obs[pert_col].astype(str) == pert) & (obs[cell_type_key].astype(str) == ct)).sum()
                )
                if ct != "__all__"
                else int((obs[pert_col].astype(str) == pert).sum()),
                "pearson_profile": r_prof,
                "pearson_delta": r_delta,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / "ifnb_pathway_real_vs_sim_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    cxp_plots.apply_style(CELLX_ROOT)
    n_ct = len(cell_types)
    ncols = 3
    nrows = int(np.ceil((n_ct + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    panels = [(ct, ct) for ct in cell_types] + [("__all__", "all IFN-beta cells")]
    for i, (key, title) in enumerate(panels):
        row = metrics_df.loc[metrics_df["cell_type"] == title]
        if row.empty and key == "__all__":
            row = metrics_df.iloc[-1:]
        m = {
            "pearson_profile": float(row["pearson_profile"].iloc[0]),
            "pearson_delta": float(row["pearson_delta"].iloc[0]),
        }
        _scatter_panel(axes_flat[i], real_pb[key], sim_pb[key], title, metrics=m)

    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Kang CollecTRI · {pert}: pseudobulk pathway scores (real vs simulated)",
        y=1.02,
        fontsize=11,
    )
    cxp_plots.save_figure(
        fig,
        f"pathway_real_vs_sim_{_safe(pert)}_facets",
        out_dir,
        description="IFN-beta pseudobulk real vs decoded simulated pathway scores per cell type",
        flat=True,
    )

    # Single-panel overview (all cells pooled)
    pooled = metrics_df[metrics_df["cell_type"] == "all IFN-beta cells"].iloc[0]
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    _scatter_panel(
        ax2,
        real_pb["__all__"],
        sim_pb["__all__"],
        f"{pert} (pooled across cell types)",
        metrics={
            "pearson_profile": float(pooled["pearson_profile"]),
            "pearson_delta": float(pooled["pearson_delta"]),
        },
    )
    cxp_plots.save_figure(
        fig2,
        f"pathway_real_vs_sim_{_safe(pert)}_pooled",
        out_dir,
        description="IFN-beta pooled pseudobulk pathway real vs simulated",
        flat=True,
    )

    summary = {
        "perturbation": pert,
        "n_pathways": meta["n_pathways"],
        "n_cells_total": meta["n_cells"],
        "metrics_csv": str(metrics_path),
        "pearson_profile_pooled": float(pooled["pearson_profile"]),
        "pearson_delta_pooled": float(pooled["pearson_delta"]),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
