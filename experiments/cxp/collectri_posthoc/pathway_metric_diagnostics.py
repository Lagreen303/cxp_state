#!/usr/bin/env python3
"""Diagnostics (1–3): latent vs pathway pearson_delta, delta scatter, DE/pdex vs ranks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

STATE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STATE_ROOT / "experiments" / "cxp"))

from cxp_collectri_posthoc import (  # noqa: E402
    align_real_pathway_scores,
    build_pathway_anndata_pair,
    decode_latent_to_pathway,
    eval_dir_for_step,
    load_decoder_weight,
    load_eval_adatas,
    load_posthoc_config,
    predicted_latent_matrix,
    resolve_latest_step,
    _barcodes_from_pred_adata,
)

CELLX_ROOT = Path("/iridisfs/ddnb/Luke/cellxpathway")
sys.path.insert(0, str(CELLX_ROOT))
from cellxpathway.eval import plots as cxp_plots  # noqa: E402

_SAFE = re.compile(r"[^\w\-]+")


def _safe(s: str) -> str:
    return _SAFE.sub("_", s.strip()).strip("_")


def _align_obsm_block(
    test_h5ad: Path,
    barcodes: np.ndarray | list[str],
    key: str,
) -> np.ndarray:
    backed = ad.read_h5ad(test_h5ad, backed="r")
    if key not in backed.obsm:
        raise KeyError(f"{key} not in {test_h5ad} obsm")
    lookup = {str(x): i for i, x in enumerate(backed.obs_names.astype(str))}
    idx = [lookup[str(b)] for b in barcodes]
    idx_arr = np.asarray(idx, dtype=np.int64)
    sort_order = np.argsort(idx_arr)
    sorted_idx = idx_arr[sort_order]
    block = np.asarray(backed.obsm[key][sorted_idx], dtype=np.float32)
    inv = np.empty_like(sort_order)
    inv[sort_order] = np.arange(sort_order.shape[0])
    out = block[inv]
    backed.file.close()
    return out


def _pseudobulk_delta(
    x: np.ndarray,
    obs: pd.DataFrame,
    pert_col: str,
    control: str,
    cell_type: str,
    cell_type_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    mask_ct = obs[cell_type_key].astype(str) == cell_type
    if not mask_ct.any():
        raise ValueError(f"no cells for {cell_type}")
    perts = obs.loc[mask_ct, pert_col].astype(str)
    x_ct = x[mask_ct.to_numpy()]
    ctrl_mask = perts == control
    if not ctrl_mask.any():
        raise ValueError(f"no control {control} in {cell_type}")
    ctrl_mean = x_ct[ctrl_mask.to_numpy()].mean(axis=0)
    rows = {}
    for pert in perts.unique():
        if pert == control:
            continue
        pm = (perts == pert).to_numpy()
        rows[pert] = x_ct[pm].mean(axis=0) - ctrl_mean
    return ctrl_mean, rows


def pearson_delta_table(
    real_x: np.ndarray,
    pred_x: np.ndarray,
    obs: pd.DataFrame,
    *,
    pert_col: str,
    control: str,
    cell_type_key: str,
    label: str,
) -> pd.DataFrame:
    cell_types = sorted(obs[cell_type_key].astype(str).unique())
    out = []
    for ct in cell_types:
        try:
            _, real_d = _pseudobulk_delta(real_x, obs, pert_col, control, ct, cell_type_key)
            _, pred_d = _pseudobulk_delta(pred_x, obs, pert_col, control, ct, cell_type_key)
        except ValueError:
            continue
        for pert in sorted(real_d.keys()):
            if pert not in pred_d:
                continue
            r = real_d[pert]
            p = pred_d[pert]
            if not (np.isfinite(r).all() and np.isfinite(p).all()):
                continue
            if np.std(r) < 1e-12 or np.std(p) < 1e-12:
                rho = np.nan
            else:
                rho = float(pearsonr(r, p)[0])
            out.append(
                {
                    "readout": label,
                    "cell_type": ct,
                    "perturbation": pert,
                    "pearson_delta": rho,
                    "var_real": float(np.var(r)),
                    "var_pred": float(np.var(p)),
                    "var_ratio_pred_over_real": float(np.var(p) / (np.var(r) + 1e-12)),
                }
            )
    return pd.DataFrame(out)


def decoder_sanity(
    real_pathway: np.ndarray,
    real_z: np.ndarray,
    decoder_w: np.ndarray,
) -> dict[str, Any]:
    decoded = decode_latent_to_pathway(real_z, decoder_w)
    # Per-cell Pearson across pathways
    n = min(5000, real_pathway.shape[0])
    rng = np.random.default_rng(42)
    sub = rng.choice(real_pathway.shape[0], size=n, replace=False)
    per_cell = []
    for i in sub:
        if np.std(real_pathway[i]) < 1e-12 or np.std(decoded[i]) < 1e-12:
            continue
        per_cell.append(float(pearsonr(real_pathway[i], decoded[i])[0]))
    flat_r = float(pearsonr(real_pathway.ravel(), decoded.ravel())[0])
    return {
        "n_cells_sampled": int(len(per_cell)),
        "per_cell_pearson_mean": float(np.mean(per_cell)) if per_cell else None,
        "per_cell_pearson_median": float(np.median(per_cell)) if per_cell else None,
        "flattened_pearson": flat_r,
        "rmse": float(np.sqrt(np.mean((real_pathway - decoded) ** 2))),
    }


def rank_abs_delta(
    real_x: np.ndarray,
    pred_x: np.ndarray,
    obs: pd.DataFrame,
    *,
    pert_col: str,
    control: str,
    cell_type: str,
    cell_type_key: str,
    perturbation: str,
) -> float:
    _, real_d = _pseudobulk_delta(real_x, obs, pert_col, control, cell_type, cell_type_key)
    _, pred_d = _pseudobulk_delta(pred_x, obs, pert_col, control, cell_type, cell_type_key)
    r = np.abs(real_d[perturbation])
    p = np.abs(pred_d[perturbation])
    if np.std(r) < 1e-12 or np.std(p) < 1e-12:
        return float("nan")
    return float(spearmanr(r, p).correlation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(STATE_ROOT / "experiments/cxp/configs/cxp_collectri_posthoc.yaml"),
    )
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument(
        "--exemplar-cell-type",
        default="CD14 Mono",
        help="Cell type for example delta scatter (item 2).",
    )
    parser.add_argument(
        "--exemplar-perturbation",
        default="IFN-gamma",
    )
    args = parser.parse_args()

    cfg = load_posthoc_config(args.config)
    run_dir = Path(cfg["run_dir"])
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    elif cfg.get("eval_dir"):
        eval_dir = Path(cfg["eval_dir"])
    else:
        step = args.checkpoint_step if args.checkpoint_step is not None else resolve_latest_step(run_dir)
        eval_dir = eval_dir_for_step(run_dir, step)
    step = args.checkpoint_step if args.checkpoint_step is not None else int(cfg.get("checkpoint_step", 0))
    if not cfg.get("infer_pred_h5ad") and step == 0:
        step = resolve_latest_step(run_dir)
    out_dir = eval_dir / "pathway_metric_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pert_col = str(cfg["pert_col"])
    cell_type_key = str(cfg["cell_type_key"])
    control = str(cfg["control_pert"])
    embed_key = str(cfg["embed_key"])
    barcode_key = str(cfg["barcode_obs_key"])
    test_h5ad = Path(cfg["test_h5ad"])
    cx_root = Path(cfg["cellxpathway_root"])

    adata_pred_path, adata_real_path, _meta = build_pathway_anndata_pair(cfg, eval_dir, checkpoint_step=step)
    obs = adata_pred_path.obs
    real_path = np.asarray(adata_real_path.X, dtype=np.float64)
    pred_path = np.asarray(adata_pred_path.X, dtype=np.float64)

    if cfg.get("infer_pred_h5ad"):
        adata_pred_lat = ad.read_h5ad(Path(cfg["infer_pred_h5ad"]))
    else:
        adata_pred_lat, _ = load_eval_adatas(eval_dir)
    barcodes = _barcodes_from_pred_adata(adata_pred_lat, barcode_key)
    z_pred = np.asarray(predicted_latent_matrix(adata_pred_lat, embed_key), dtype=np.float64)
    z_real = _align_obsm_block(test_h5ad, barcodes, embed_key).astype(np.float64)

    dec_w = load_decoder_weight(
        cx_root,
        cx_root / cfg["encode_config"],
        n_path=int(real_path.shape[1]),
        decoder_weight_npy=cfg.get("decoder_weight_npy"),
    )
    sanity = decoder_sanity(real_path.astype(np.float32), z_real.astype(np.float32), dec_w)

    tbl_latent = pearson_delta_table(
        z_real, z_pred, obs, pert_col=pert_col, control=control, cell_type_key=cell_type_key, label="latent_X_cxp"
    )
    tbl_path = pearson_delta_table(
        real_path,
        pred_path,
        obs,
        pert_col=pert_col,
        control=control,
        cell_type_key=cell_type_key,
        label="pathway_decoded",
    )
    decoded_from_real_z = decode_latent_to_pathway(z_real.astype(np.float32), dec_w).astype(np.float64)
    tbl_obs_vs_decode_real_z = pearson_delta_table(
        real_path,
        decoded_from_real_z,
        obs,
        pert_col=pert_col,
        control=control,
        cell_type_key=cell_type_key,
        label="pathway_observed_vs_decode_real_z",
    )
    decompose = pd.concat([tbl_latent, tbl_path, tbl_obs_vs_decode_real_z], ignore_index=True)
    decompose_path = out_dir / "decompose_pearson_delta.csv"
    decompose.to_csv(decompose_path, index=False)

    # Item 2: exemplar delta scatter + variance summary
    ct_ex = str(cfg.get("exemplar_cell_type", args.exemplar_cell_type))
    pert_ex = str(cfg.get("exemplar_perturbation", args.exemplar_perturbation))
    _, real_d = _pseudobulk_delta(real_path, obs, pert_col, control, ct_ex, cell_type_key)
    _, pred_d = _pseudobulk_delta(pred_path, obs, pert_col, control, ct_ex, cell_type_key)
    if pert_ex in real_d and pert_ex in pred_d:
        r = real_d[pert_ex]
        p = pred_d[pert_ex]

        cxp_plots.apply_style(CELLX_ROOT)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(r, p, s=8, alpha=0.5, color=cxp_plots.CATEGORICAL_HUES[0], edgecolors="none")
        lim_lo = float(min(r.min(), p.min()))
        lim_hi = float(max(r.max(), p.max()))
        pad = 0.05 * (lim_hi - lim_lo + 1e-9)
        lim = (lim_lo - pad, lim_hi + pad)
        ax.plot(lim, lim, color="0.5", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("real Δ pathway")
        ax.set_ylabel("pred Δ pathway")
        pr = pearsonr(r, p)[0] if np.std(r) > 1e-12 and np.std(p) > 1e-12 else float("nan")
        ax.set_title(f"{ct_ex} · {pert_ex}\npearson_delta={pr:.3f}, var ratio={np.var(p)/(np.var(r)+1e-12):.3f}")
        ax.grid(True, alpha=0.25)
        cxp_plots.save_figure(
            fig,
            f"delta_scatter_{_safe(ct_ex)}_{_safe(pert_ex)}",
            out_dir,
            description="Real vs pred pathway pseudobulk delta",
            flat=True,
        )

    var_summary = tbl_path.groupby("perturbation")[["var_ratio_pred_over_real", "pearson_delta"]].agg(["mean", "median"])
    var_summary.to_csv(out_dir / "pathway_var_ratio_by_perturbation.csv")

    # Item 3: DE nsig + rank correlation vs pr_auc from cell-eval results
    celleval_dir = eval_dir / "pathway_celleval"
    de_rows = []
    rank_rows = []
    for path in sorted(celleval_dir.glob("*_results.csv")):
        if path.name.endswith("_agg_results.csv"):
            continue
        ct = path.stem.replace("_results", "")
        df = pd.read_csv(path)
        if "perturbation" not in df.columns:
            continue
        for _, row in df.iterrows():
            pert = str(row["perturbation"])
            de_rows.append(
                {
                    "cell_type": ct,
                    "perturbation": pert,
                    "de_nsig_counts_real": row.get("de_nsig_counts_real"),
                    "de_nsig_counts_pred": row.get("de_nsig_counts_pred"),
                    "pr_auc": row.get("pr_auc"),
                    "pearson_delta_celleval": row.get("pearson_delta"),
                }
            )
            try:
                rho_rank = rank_abs_delta(
                    real_path,
                    pred_path,
                    obs,
                    pert_col=pert_col,
                    control=control,
                    cell_type=ct,
                    cell_type_key=cell_type_key,
                    perturbation=pert,
                )
            except (ValueError, KeyError):
                rho_rank = float("nan")
            rank_rows.append(
                {
                    "cell_type": ct,
                    "perturbation": pert,
                    "spearman_abs_delta": rho_rank,
                    "pr_auc": row.get("pr_auc"),
                    "de_nsig_counts_pred": row.get("de_nsig_counts_pred"),
                }
            )

    de_df = pd.DataFrame(de_rows)
    rank_df = pd.DataFrame(rank_rows)
    de_df.to_csv(out_dir / "de_pdex_vs_metrics.csv", index=False)
    rank_df.to_csv(out_dir / "rank_abs_delta_vs_pr_auc.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax0, ax1 = axes
    if not de_df.empty:
        ax0.scatter(
            de_df["de_nsig_counts_real"],
            de_df["de_nsig_counts_pred"],
            c=de_df["pr_auc"],
            cmap="viridis",
            s=18,
            alpha=0.85,
        )
        ax0.set_xlabel("de_nsig_counts_real")
        ax0.set_ylabel("de_nsig_counts_pred")
        ax0.set_title("pdex significant pathway counts")
    if not rank_df.empty:
        sub = rank_df.dropna(subset=["spearman_abs_delta", "pr_auc"])
        ax1.scatter(sub["spearman_abs_delta"], sub["pr_auc"], s=18, alpha=0.85, color=cxp_plots.CATEGORICAL_HUES[1])
        ax1.set_xlabel("spearman(|Δ_real|, |Δ_pred|)")
        ax1.set_ylabel("pr_auc (cell-eval)")
        ax1.set_title("Rank agreement vs PR-AUC")
    fig.tight_layout()
    cxp_plots.save_figure(
        fig,
        "de_pdex_and_rank_vs_pr_auc",
        out_dir,
        description="DE nsig counts and rank correlation vs pr_auc",
        flat=True,
    )

    # Latent vs pathway pearson_delta comparison
    wide = tbl_latent.merge(
        tbl_path,
        on=["cell_type", "perturbation"],
        suffixes=("_latent", "_pathway"),
    )
    if not wide.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            wide["pearson_delta_latent"],
            wide["pearson_delta_pathway"],
            s=20,
            alpha=0.8,
            color=cxp_plots.CATEGORICAL_HUES[0],
        )
        lo = float(min(wide["pearson_delta_latent"].min(), wide["pearson_delta_pathway"].min()))
        hi = float(max(wide["pearson_delta_latent"].max(), wide["pearson_delta_pathway"].max()))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.plot((lo - pad, hi + pad), (lo - pad, hi + pad), color="0.5", lw=0.8)
        ax.set_xlabel("pearson_delta latent (X_cxp)")
        ax.set_ylabel("pearson_delta pathway (decoded)")
        ax.set_title("Readout decomposition")
        ax.grid(True, alpha=0.25)
        cxp_plots.save_figure(
            fig,
            "pearson_delta_latent_vs_pathway",
            out_dir,
            description="Latent vs pathway pearson_delta",
            flat=True,
        )

    pred_zero = rank_df[rank_df["de_nsig_counts_pred"].fillna(0) == 0]
    report = {
        "checkpoint_step": step,
        "eval_dir": str(eval_dir),
        "decoder_sanity_real_z": sanity,
        "pearson_delta_mean": {
            "latent_X_cxp": float(tbl_latent["pearson_delta"].mean()),
            "pathway_decoded": float(tbl_path["pearson_delta"].mean()),
        },
        "n_pairs": {
            "latent": int(len(tbl_latent)),
            "pathway": int(len(tbl_path)),
        },
        "exemplar": {"cell_type": ct_ex, "perturbation": pert_ex, "pearson_delta": float(pr)},
        "de_nsig_pred_zero_fraction": float((de_df["de_nsig_counts_pred"].fillna(0) == 0).mean())
        if not de_df.empty
        else None,
        "when_pred_nsig_zero": {
            "spearman_abs_delta_mean": float(pred_zero["spearman_abs_delta"].mean()) if len(pred_zero) else None,
            "pr_auc_mean": float(pred_zero["pr_auc"].mean()) if len(pred_zero) else None,
            "n": int(len(pred_zero)),
        },
        "outputs": {
            "decompose_pearson_delta_csv": str(decompose_path),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
