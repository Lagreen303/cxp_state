#!/usr/bin/env python3
"""
Generate n-cells-per-cell-type-and-condition reports.

Atlas: use existing report at /iridisfs/ddnb/Luke/data/parse-10M-90cytokines/parse_VCC_report.txt
GSE96583, GSE178429: generated here → biological_coherence/reports/
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

BASE = Path("/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026")
OUTDIR = BASE / "biological_coherence" / "reports"

# Atlas: use existing report at /iridisfs/ddnb/Luke/data/parse-10M-90cytokines/parse_VCC_report.txt
ATLAS_REPORT = Path("/iridisfs/ddnb/Luke/data/parse-10M-90cytokines/parse_VCC_report.txt")

DATASETS = {
    "GSE96583": {
        "path": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/clustering/GSE96583_batch2_IFNbeta_clustered_manual_annotated.h5ad"),
        "fallback": Path("/iridisfs/ddnb/Luke/data/GSE96583_Kang/GSE96583_batch2_IFNbeta_aligned_avg.h5ad"),
        "ct_col": "cell_type",
        "condition_col": "cytokine",
    },
    "GSE178429": {
        "path": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg.h5ad"),
        "fallback": Path("/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_IFN_Control_1h6h_aligned_zero.h5ad"),
        "ct_col": "cell_type",
        "condition_col": "cytokine",  # or StimType
    },
}


def _resolve_condition_col(adata: ad.AnnData, preferred: str) -> str | None:
    for col in [preferred, "treatment", "cytokine", "StimType", "condition"]:
        if col in adata.obs.columns:
            return col
    return None


def write_report(ds_name: str, adata: ad.AnnData, ct_col: str, cond_col: str, outpath: Path) -> None:
    obs = adata.obs
    lines = [
        f"=== {ds_name} cell count report ===",
        f"Source: {adata.n_obs} cells × {adata.n_vars} genes",
        "",
        "--- By cell type ---",
    ]
    ct_counts = obs[ct_col].astype(str).value_counts().sort_index()
    for ct, n in ct_counts.items():
        lines.append(f"  {ct}: {n:,}")
    lines.append(f"\n  Total: {ct_counts.sum():,}")

    lines.extend(["", "--- By condition (cytokine/treatment) ---"])
    cond_counts = obs[cond_col].astype(str).value_counts().sort_index()
    for c, n in cond_counts.items():
        lines.append(f"  {c}: {n:,}")
    lines.append(f"\n  Total: {cond_counts.sum():,}")

    # Cross-tab: n cells per (cell_type, condition)
    lines.extend(["", "--- n cells per cell type × condition ---"])
    ct = pd.Series(obs[ct_col].astype(str))
    cond = pd.Series(obs[cond_col].astype(str))
    cross = pd.crosstab(ct, cond)
    cross.index.name = "cell_type"
    cross.columns.name = "condition"
    lines.append(cross.to_string())
    lines.append(f"\nTotal: {cross.sum().sum():,}")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {outpath}")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Generating cell count reports for 3 datasets...\n")

    for ds_name, cfg in DATASETS.items():
        path = cfg["path"]
        if not path.exists() and "fallback" in cfg:
            path = cfg["fallback"]
        if not path.exists():
            print(f"[SKIP] {ds_name}: {path} not found")
            continue

        print(f"Loading {ds_name}...")
        adata = ad.read_h5ad(path)
        ct_col = cfg["ct_col"]
        preferred_cond = cfg.get("condition_col", "cytokine")

        if ct_col not in adata.obs.columns:
            print(f"  [WARN] {ds_name}: no '{ct_col}' column. Available: {list(adata.obs.columns)}")
            # Try common alternatives
            for alt in ["celltype", "cell_type", "cluster"]:
                if alt in adata.obs.columns:
                    ct_col = alt
                    break
            else:
                print(f"  [SKIP] {ds_name}: cannot find cell type column")
                continue

        cond_col = _resolve_condition_col(adata, preferred_cond)
        if cond_col is None:
            print(f"  [WARN] {ds_name}: no condition column. Using 'cytokine' if present else skip.")
            cond_col = "cytokine" if "cytokine" in adata.obs.columns else None
        if cond_col is None:
            print(f"  [SKIP] {ds_name}: no condition/cytokine column. Available: {list(adata.obs.columns)}")
            continue

        outpath = OUTDIR / f"cell_counts_{ds_name}.txt"
        write_report(ds_name, adata, ct_col, cond_col, outpath)
        del adata

    print("\nDone. All 3 reports:")
    print(f"  Atlas:      {ATLAS_REPORT}")
    print(f"  GSE96583:   {OUTDIR / 'cell_counts_GSE96583.txt'}")
    print(f"  GSE178429:  {OUTDIR / 'cell_counts_GSE178429.txt'}")


if __name__ == "__main__":
    main()
