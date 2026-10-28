#!/usr/bin/env python3
"""UMAP panels: observed vs decoded pathway scores (CollecTRI posthoc)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STATE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STATE_ROOT / "experiments" / "cxp"))

from cxp_collectri_posthoc import (  # noqa: E402
    build_pathway_anndata_pair,
    eval_dir_for_step,
    load_posthoc_config,
    resolve_latest_step,
    subsample_indices,
)


def main() -> None:
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
    run_tag = str(cfg.get("eval_label", f"step{step}"))

    cx_root = Path(cfg["cellxpathway_root"])
    sys.path.insert(0, str(cx_root))

    from cellxpathway.eval.e0c_metrics import compute_score_umap
    from cellxpathway.eval.e2_grid_function_latent import stim_vs_control_umap_labels
    from cellxpathway.eval.plots import (
        apply_style,
        celltype_umap_scatter,
        save_figure,
        scfm_distinct_umap_scatter,
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    adata_pred, adata_real, meta = build_pathway_anndata_pair(cfg, eval_dir, checkpoint_step=step)
    n = adata_pred.n_obs
    max_cells = int(cfg.get("max_umap_cells", 80000))
    seed = int(cfg.get("seed", 42))
    sub = subsample_indices(n, max_cells, seed)

    pred_sub = adata_pred[sub].copy()
    real_sub = adata_real[sub].copy()

    out_dir = eval_dir / "figures_umap"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style(cx_root)

    pert_col = str(cfg["pert_col"])
    stem_prefix = f"pathway_{run_tag}"

    for which, adata_sub in (("real", real_sub), ("pred", pred_sub)):
        mat = np.asarray(adata_sub.X, dtype=np.float64)
        umap = compute_score_umap(mat, seed=seed)
        cell_types = adata_sub.obs[str(cfg["cell_type_key"])].astype(str).to_numpy()
        fig_ct = celltype_umap_scatter(umap, cell_types)
        fig_ct.suptitle(f"Pathway {which} ({run_tag}): UMAP by cell_type")
        png_ct, pdf_ct = save_figure(
            fig_ct,
            f"{stem_prefix}_{which}_cell_type",
            out_dir,
            description=f"CollecTRI pathway {which} ({run_tag}), cell_type",
            flat=True,
        )
        cytokines = adata_sub.obs[pert_col].astype(str).to_numpy()
        fig_cy = scfm_distinct_umap_scatter(
            umap,
            cytokines,
            legend_title="Cytokine",
            legend_ncol=2,
        )
        fig_cy.suptitle(f"Pathway {which} ({run_tag}): UMAP by cytokine")
        png_cy, pdf_cy = save_figure(
            fig_cy,
            f"{stem_prefix}_{which}_cytokine",
            out_dir,
            description=f"CollecTRI pathway {which} ({run_tag}), cytokine",
            flat=True,
        )
        stim_labels = stim_vs_control_umap_labels(cytokines, control=str(cfg["control_pert"]))
        fig_stim = scfm_distinct_umap_scatter(
            umap,
            stim_labels,
            legend_title="Condition",
            legend_ncol=1,
        )
        fig_stim.suptitle(f"Pathway {which} ({run_tag}): control vs stimulated")
        png_stim, pdf_stim = save_figure(
            fig_stim,
            f"{stem_prefix}_{which}_condition",
            out_dir,
            description=f"CollecTRI pathway {which} ({run_tag}), control vs stim",
            flat=True,
        )
        meta[f"figures_{which}"] = {
            "cell_type": [str(png_ct), str(pdf_ct)],
            "cytokine": [str(png_cy), str(pdf_cy)],
            "condition": [str(png_stim), str(pdf_stim)],
        }

    manifest = {
        "kind": "cxp_collectri_pathway_umap",
        "checkpoint_step": step,
        "eval_dir": str(eval_dir),
        "n_cells_total": n,
        "n_cells_umap": int(sub.shape[0]),
        **meta,
    }
    man_path = out_dir / "umap_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[INFO] Wrote {man_path}")


if __name__ == "__main__":
    main()
