#!/usr/bin/env python3
"""Frozen VAE decode + cell-eval full profile on pathway readout space."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STATE_ROOT / "experiments" / "cxp"))

from cxp_collectri_posthoc import (  # noqa: E402
    build_pathway_anndata_pair,
    eval_dir_for_step,
    load_posthoc_config,
    resolve_latest_step,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(STATE_ROOT / "experiments" / "cxp/configs/cxp_collectri_posthoc.yaml"),
    )
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--eval-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_posthoc_config(args.config)
    run_dir = Path(cfg["run_dir"])
    step = args.checkpoint_step if args.checkpoint_step is not None else resolve_latest_step(run_dir)
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    elif cfg.get("eval_dir"):
        eval_dir = Path(cfg["eval_dir"])
    else:
        suffix = str(cfg.get("eval_dir_suffix", ""))
        eval_dir = eval_dir_for_step(run_dir, step, suffix=suffix)

    adata_pred, adata_real, meta = build_pathway_anndata_pair(cfg, eval_dir, checkpoint_step=step)

    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype

    outdir = eval_dir / "pathway_celleval"
    outdir.mkdir(parents=True, exist_ok=True)

    pert_col = str(cfg["pert_col"])
    cell_type_key = str(cfg["cell_type_key"])
    control_pert = str(cfg["control_pert"])

    ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=cell_type_key)
    ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=cell_type_key)
    assert len(ct_split_real) == len(ct_split_pred), (
        f"cell type count mismatch: {len(ct_split_real)} != {len(ct_split_pred)}"
    )

    pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
    # Pathway scores are z-scored readouts, not log1p gene counts.
    allow_discrete = True
    manifest_ct: dict[str, str] = {}
    skipped_ct: dict[str, str] = {}

    for ct in ct_split_real.keys():
        real_ct = ct_split_real[ct]
        pred_ct = ct_split_pred[ct]
        perts_real = set(real_ct.obs[pert_col].astype(str).unique())
        perts_pred = set(pred_ct.obs[pert_col].astype(str).unique())
        if control_pert not in perts_real or control_pert not in perts_pred:
            reason = (
                f"no {control_pert} controls "
                f"(real={sorted(perts_real)}, pred={sorted(perts_pred)})"
            )
            skipped_ct[ct] = reason
            logger.warning("Skipping cell type %s: %s", ct, reason)
            continue
        evaluator = MetricsEvaluator(
            adata_pred=pred_ct,
            adata_real=real_ct,
            control_pert=control_pert,
            pert_col=pert_col,
            outdir=str(outdir),
            prefix=ct,
            pdex_kwargs=pdex_kwargs,
            batch_size=2048,
            allow_discrete=allow_discrete,
        )
        evaluator.compute(
            profile="full",
            skip_metrics=["pearson_edistance", "clustering_agreement"],
        )
        manifest_ct[ct] = "ok"

    pred_path = outdir / "adata_pred_pathway.h5ad"
    real_path = outdir / "adata_real_pathway.h5ad"
    adata_pred.write_h5ad(pred_path)
    adata_real.write_h5ad(real_path)

    manifest = {
        "kind": "cxp_collectri_pathway_celleval",
        "checkpoint_step": step,
        "eval_dir": str(eval_dir),
        "outdir": str(outdir),
        "allow_discrete": allow_discrete,
        "profile": "full",
        "cell_types": manifest_ct,
        "skipped_cell_types": skipped_ct,
        **meta,
    }
    man_path = outdir / "celleval_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[INFO] Wrote metrics under {outdir}")
    print(f"[INFO] Wrote {man_path}")


if __name__ == "__main__":
    main()
