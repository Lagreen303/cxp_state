#!/usr/bin/env python3
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

"""
Generic single-cell clustering script for .h5ad files.

Pipeline:
  - (Optional) subset by obs column and values
  - Normalize total counts per cell
  - log1p
  - Find highly variable genes
  - PCA
  - Neighbor graph
  - UMAP
  - Leiden clustering

Outputs:
  - Clustered AnnData written to --output (.h5ad)
  - UMAP PNG saved as <output_basename>_umap.png in the same directory

Example:
  uv run --with 'scanpy,anndata' python data/wrangling/cluster_h5ad.py \\
    --input /iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_stim_scRNAseq.h5ad \\
    --output /iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_stim_scRNAseq_clustered.h5ad

Optional subset to specific conditions:
  --subset-col Condition --subset-values Control_1h Control_6h IFN_1h IFN_6h
"""

import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster a single-cell .h5ad and save UMAP PNG.")
    p.add_argument("--input", required=True, help="Input .h5ad file")
    p.add_argument("--output", required=True, help="Output .h5ad file (clustered)")

    p.add_argument(
        "--subset-col",
        type=str,
        default=None,
        help="Optional obs column to subset on (e.g. Condition, StimType)",
    )
    p.add_argument(
        "--subset-values",
        nargs="+",
        default=None,
        help="Values to keep in subset-col (space-separated)",
    )
    p.add_argument(
        "--n-hvgs",
        type=int,
        default=2000,
        help="Number of highly variable genes (default: 2000)",
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Leiden resolution (default: 0.5)",
    )
    p.add_argument(
        "--umap-color",
        type=str,
        default="leiden",
        help="obs column to color UMAP by (default: 'leiden')",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...", flush=True)
    adata = ad.read_h5ad(in_path)
    print(f"  shape: {adata.shape}", flush=True)

    # Optional subset
    if args.subset_col is not None and args.subset_values:
        if args.subset_col not in adata.obs.columns:
            raise KeyError(f"subset-col '{args.subset_col}' not found in adata.obs")
        mask = adata.obs[args.subset_col].isin(args.subset_values)
        print(
            f"Subsetting on {args.subset_col} in {args.subset_values}: "
            f"{mask.sum()} / {adata.n_obs} cells kept",
            flush=True,
        )
        adata = adata[mask].copy()

    # Basic Scanpy pipeline
    # Manual normalize_total to avoid rare scanpy bug with counts_per_cols
    import numpy as np
    from scipy import sparse as sp

    print("Normalizing total counts per cell to 1e4 ...", flush=True)
    X = adata.X
    if sp.issparse(X):
        # work in float to avoid integer casting issues
        X = X.tocsr().astype("float32")
        counts = np.asarray(X.sum(axis=1)).ravel()
        counts[counts == 0] = 1.0
        scaling = 1e4 / counts
        # scale rows in-place
        X.data *= np.repeat(scaling, np.diff(X.indptr))
        adata.X = X
    else:
        counts = X.sum(axis=1)
        counts[counts == 0] = 1.0
        scaling = 1e4 / counts
        adata.X = (X.T * scaling).T
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvgs, subset=True)
    print(f"HVGs selected: {adata.n_vars}", flush=True)

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution=args.resolution, key_added="leiden")
    print("Leiden clusters:", adata.obs["leiden"].value_counts().to_dict(), flush=True)

    # Save clustered AnnData
    print(f"Writing clustered AnnData to {out_path} ...", flush=True)
    adata.write_h5ad(out_path)

    # Save UMAP PNG
    umap_png = out_path.with_suffix("")  # strip .h5ad
    umap_png = umap_png.with_name(umap_png.name + "_umap.png")
    print(f"Saving UMAP to {umap_png} ...", flush=True)
    sc.pl.umap(adata, color=args.umap_color, show=False, save=None)
    # By default, sc.pl.umap saves to a global sc.settings.figdir; instead:
    # use matplotlib directly to control exact path.
    import matplotlib.pyplot as plt

    plt.savefig(umap_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

