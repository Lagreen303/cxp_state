#!/usr/bin/env python3
"""
Compute intersection of genes across multiple AnnData .h5ad files WITHOUT loading X.

Supports PER-DATASET gene-name location:
- use adata.var_names
- OR a specified adata.var column (e.g. "feature_name")

Writes the common genes in the REFERENCE dataset's ordering.
"""

import anndata as ad
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

# Reference dataset defines the preferred output ordering
REF_PATH = "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/Parse_10M_PBMC_cytokines.h5ad"

# Per-dataset gene-name source:
#   - None => use adata.var_names
#   - "feature_name" (or any string) => use adata.var[that_column]
GENE_NAME_SOURCE = {
    REF_PATH: None,  # <-- set per dataset
    "/iridisfs/ddnb/Luke/data/Onek1k/2a930352-0802-4af7-bd26-61f75c83e1b0/eQTLAutoimmune.h5ad": "feature_name",
    # Add more datasets here:
    # "/path/to/other.h5ad": None,
    # "/path/to/another.h5ad": "feature_name",
}

OUTPUT_GENES = "/iridisfs/ddnb/Luke/data/Onek1k/cytokine_1k1k_parse_common_genes.txt"

# Optional: If a dataset path isn't in GENE_NAME_SOURCE, choose behaviour:
#   - "error" => raise KeyError
#   - "var_names" => default to var_names
#   - "feature_name" => default to var["feature_name"]
DEFAULT_IF_MISSING = "error"

# Optional: If gene source is a var column, drop NA/empty values
DROP_EMPTY = True

# ============================================================
# END CONFIG
# ============================================================


def get_gene_names(adata: ad.AnnData, gene_col: str | None, dataset_path: str) -> list[str]:
    """Extract gene names using either var_names or a specified var column."""
    if gene_col is None:
        genes = adata.var_names.astype(str).tolist()
    else:
        if gene_col not in adata.var.columns:
            raise KeyError(
                f"[ERROR] Dataset: {dataset_path}\n"
                f"Gene column '{gene_col}' not found in adata.var.\n"
                f"Available columns: {list(adata.var.columns)}"
            )
        genes = adata.var[gene_col].astype(str).tolist()

    if DROP_EMPTY:
        genes = [g for g in genes if g and g.lower() != "nan"]

    if not genes:
        raise RuntimeError(f"[ERROR] No valid gene names extracted for dataset: {dataset_path}")

    return genes


def resolve_gene_source(dataset_path: str) -> str | None:
    """Resolve per-dataset gene column choice."""
    if dataset_path in GENE_NAME_SOURCE:
        return GENE_NAME_SOURCE[dataset_path]

    if DEFAULT_IF_MISSING == "error":
        raise KeyError(
            f"[ERROR] No gene-name source configured for dataset:\n  {dataset_path}\n"
            f"Add it to GENE_NAME_SOURCE, or set DEFAULT_IF_MISSING."
        )
    if DEFAULT_IF_MISSING == "var_names":
        return None
    if DEFAULT_IF_MISSING == "feature_name":
        return "feature_name"

    raise ValueError(f"[ERROR] DEFAULT_IF_MISSING must be one of: error, var_names, feature_name")


def main():
    datasets = list(GENE_NAME_SOURCE.keys())
    if REF_PATH not in datasets:
        raise ValueError("[ERROR] REF_PATH must be included as a key in GENE_NAME_SOURCE")

    # Ensure reference is first (ordering)
    datasets = [REF_PATH] + [p for p in datasets if p != REF_PATH]

    gene_sets = []
    ref_order = None

    for i, p in enumerate(datasets):
        print(f"[INFO] Reading genes from: {p}", flush=True)
        a = ad.read_h5ad(p, backed="r")

        gene_col = resolve_gene_source(p)
        src = "var_names" if gene_col is None else f"var['{gene_col}']"
        print(f"[INFO] Gene source: {src}", flush=True)

        genes = get_gene_names(a, gene_col, p)
        gene_set = set(genes)

        print(f"[INFO] Unique genes: {len(gene_set):,}", flush=True)

        # Save reference ordering (first dataset only)
        if i == 0:
            ref_order = genes

        gene_sets.append(gene_set)

    common = set.intersection(*gene_sets)
    print(f"[INFO] Genes present in ALL datasets: {len(common):,}", flush=True)

    if not common:
        raise RuntimeError("[ERROR] Intersection is empty. Check gene naming conventions and sources.")

    # Write in reference order
    out_path = Path(OUTPUT_GENES)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    with open(out_path, "w") as f:
        for g in ref_order:
            if g in common:
                f.write(g + "\n")
                wrote += 1

    print(f"[OK] Wrote {wrote:,} common genes to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
