"""Shared helpers: CollecTRI latent predict → pathway space for UMAP and cell-eval."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

_STEP_RE = re.compile(r"^step=(\d+)\.ckpt$")


def load_posthoc_config(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_latest_step(run_dir: Path) -> int:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")
    best_step = -1
    for p in ckpt_dir.iterdir():
        if not p.is_file() or not p.name.endswith(".ckpt"):
            continue
        m = _STEP_RE.match(p.name)
        if m is None:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
    if best_step < 0:
        raise FileNotFoundError(f"No step=N.ckpt under {ckpt_dir}")
    return best_step


def eval_dir_for_step(run_dir: Path, step: int, *, suffix: str = "") -> Path:
    return run_dir / f"eval_step={step}.ckpt{suffix}"


def count_nonfinite(arr: np.ndarray) -> int:
    return int(np.size(arr) - np.isfinite(arr).sum())


def assert_finite_report(arr: np.ndarray, context: str) -> None:
    n_bad = count_nonfinite(arr)
    if n_bad:
        raise ValueError(f"non-finite values in {context}: count={n_bad}")


def load_eval_adatas(eval_dir: Path) -> tuple[ad.AnnData, ad.AnnData]:
    pred_path = eval_dir / "adata_pred.h5ad"
    real_path = eval_dir / "adata_real.h5ad"
    for p in (pred_path, real_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    return ad.read_h5ad(pred_path), ad.read_h5ad(real_path)


def predicted_latent_matrix(adata_pred: ad.AnnData, embed_key: str) -> np.ndarray:
    if embed_key in adata_pred.obsm:
        z = np.asarray(adata_pred.obsm[embed_key], dtype=np.float32)
    else:
        z = np.asarray(adata_pred.X, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"predicted latent must be 2d, got {z.shape}")
    return z


def align_real_pathway_scores(
    test_h5ad: Path,
    barcodes: np.ndarray | list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    barcodes = [str(b) for b in barcodes]
    backed = ad.read_h5ad(test_h5ad, backed="r")
    lookup = {str(x): i for i, x in enumerate(backed.obs_names.astype(str))}
    idx = []
    missing = []
    for b in barcodes:
        if b not in lookup:
            missing.append(b)
        else:
            idx.append(lookup[b])
    if missing:
        raise KeyError(f"{len(missing)} barcodes missing from {test_h5ad} (first 5: {missing[:5]})")
    idx_arr = np.asarray(idx, dtype=np.int64)
    sort_order = np.argsort(idx_arr)
    sorted_idx = idx_arr[sort_order]
    x_block = backed.X[sorted_idx]
    if hasattr(x_block, "toarray"):
        scores_sorted = np.asarray(x_block.toarray(), dtype=np.float32)
    else:
        scores_sorted = np.asarray(x_block, dtype=np.float32)
    inv = np.empty_like(sort_order)
    inv[sort_order] = np.arange(sort_order.shape[0])
    scores = scores_sorted[inv]
    var = backed.var.copy()
    backed.file.close()
    assert_finite_report(scores, "real pathway scores from test h5ad")
    return scores, var


def _ensure_cellxpathway_import(cellxpathway_root: Path) -> None:
    root = str(cellxpathway_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_decoder_weight(
    cellxpathway_root: Path,
    encode_cfg_path: Path,
    n_path: int,
    *,
    decoder_weight_npy: str | None = None,
) -> np.ndarray:
    """Load frozen linear decoder W (out_features × z_dim); prefer cached .npy (no torch import)."""
    if decoder_weight_npy:
        npy_path = (cellxpathway_root / decoder_weight_npy).resolve()
        if npy_path.is_file():
            w = np.load(npy_path).astype(np.float32)
            if w.shape[0] != n_path:
                raise ValueError(f"decoder_weight rows {w.shape[0]} != n_pathways {n_path}")
            assert_finite_report(w, "decoder weight npy")
            return w

    _ensure_cellxpathway_import(cellxpathway_root)
    from pathwayperturb.data.delta_build import _linear_decoder_weight

    with open(encode_cfg_path) as f:
        enc = yaml.safe_load(f)
    root = cellxpathway_root
    fusion_exp = (root / enc["fusion_experiment"]).resolve()
    checkpoint = (root / enc["checkpoint"]).resolve()
    w = _linear_decoder_weight(checkpoint, fusion_exp, root, n_path)
    assert_finite_report(w, "decoder weight")
    return w.astype(np.float32)


def decode_latent_to_pathway(
    z: np.ndarray,
    decoder_weight: np.ndarray,
    *,
    batch_cells: int = 8192,
) -> np.ndarray:
    """Linear decoder: p_hat = z @ W.T (matches fusion PathwayDecoder linear)."""
    n = z.shape[0]
    out_dim = decoder_weight.shape[0]
    out = np.empty((n, out_dim), dtype=np.float32)
    for start in range(0, n, batch_cells):
        end = min(start + batch_cells, n)
        block = z[start:end].astype(np.float64, copy=False)
        out[start:end] = (block @ decoder_weight.T).astype(np.float32)
    assert_finite_report(out, "decoded pathway scores")
    return out


def _barcodes_from_pred_adata(adata_pred_lat: ad.AnnData, barcode_key: str) -> np.ndarray:
    if barcode_key in adata_pred_lat.obs.columns:
        return adata_pred_lat.obs[barcode_key].astype(str).to_numpy()
    return adata_pred_lat.obs_names.astype(str).to_numpy()


def build_pathway_anndata_pair(
    cfg: dict[str, Any],
    eval_dir: Path,
    *,
    checkpoint_step: int | None = None,
) -> tuple[ad.AnnData, ad.AnnData, dict[str, Any]]:
    embed_key = str(cfg["embed_key"])
    barcode_key = str(cfg.get("barcode_obs_key", "pert_cell_barcode"))
    cx_root = Path(cfg["cellxpathway_root"])
    encode_cfg = cx_root / cfg["encode_config"]

    infer_pred = cfg.get("infer_pred_h5ad")
    if infer_pred:
        adata_pred_lat = ad.read_h5ad(Path(infer_pred))
        test_h5ad = Path(cfg.get("infer_real_h5ad") or cfg["test_h5ad"])
        checkpoint_step = int(cfg.get("checkpoint_step", 0))
    else:
        run_dir = Path(cfg["run_dir"])
        if checkpoint_step is None:
            checkpoint_step = resolve_latest_step(run_dir)
        test_h5ad = Path(cfg["test_h5ad"])
        adata_pred_lat, _adata_real_lat = load_eval_adatas(eval_dir)

    barcodes = _barcodes_from_pred_adata(adata_pred_lat, barcode_key)

    real_x, var = align_real_pathway_scores(test_h5ad, barcodes)
    z_pred = predicted_latent_matrix(adata_pred_lat, embed_key)
    if real_x.shape[0] != z_pred.shape[0]:
        raise ValueError(f"row mismatch real={real_x.shape[0]} pred={z_pred.shape[0]}")

    dec_w = load_decoder_weight(
        cx_root,
        encode_cfg,
        n_path=int(real_x.shape[1]),
        decoder_weight_npy=cfg.get("decoder_weight_npy"),
    )
    pred_x = decode_latent_to_pathway(
        z_pred,
        dec_w,
        batch_cells=int(cfg.get("decode_batch_cells", 8192)),
    )

    obs = adata_pred_lat.obs.copy()
    adata_real = ad.AnnData(X=real_x, obs=obs, var=var)
    adata_pred = ad.AnnData(X=pred_x, obs=obs.copy(), var=var)

    meta = {
        "checkpoint_step": checkpoint_step,
        "eval_dir": str(eval_dir),
        "n_cells": int(real_x.shape[0]),
        "n_pathways": int(real_x.shape[1]),
        "nonfinite_real": count_nonfinite(real_x),
        "nonfinite_pred": count_nonfinite(pred_x),
    }
    return adata_pred, adata_real, meta


def subsample_indices(n: int, max_cells: int, seed: int) -> np.ndarray:
    if n <= max_cells:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_cells, replace=False))
