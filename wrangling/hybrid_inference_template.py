"""
Build a template AnnData for STATE inference.

NEW FEATURE:
    Optional --common-genes <genes.txt> argument:
    - genes.txt contains one gene per line
    - If provided, the script will subset BOTH:
        (1) the cytokine AnnData template
        (2) each perturbation dictionary AnnData
      to those genes before assigning test perturbations.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# ============================================================
# USER CONFIG — EDIT THIS BLOCK
# ============================================================

# Paths to perturbation dictionary AnnDatas (e.g. Replogle screens)
PERT_DICT_PATHS = [
    "/iridisfs/ddnb/Luke/data/Replogle/ReplogleWeissman2022_K562_essential.h5ad",
    "/iridisfs/ddnb/Luke/data/Replogle/ReplogleWeissman2022_rpe1.h5ad",
]

# Column names
INPUT_CELLTYPE_COL = "cell_type"
INPUT_PERT_COL     = "cytokine"
DICT_PERT_COL      = "gene"

# Cell type to keep from the input AnnData
TARGET_CELLTYPE    = "CD14 Mono"

# Label marking the special control/stimulus state
CONTROL_STIM_LABEL = "PBS"

# Perturbations to exclude from dictionary sets
EXCLUDE_PERTS = {"control", "non-targeting", "NT", "ntc"}

# Column to create for test perturbations
TEST_PERT_COL = "pert_gene_test"

# Column marking control-like rows
MARK_CONTROL_STIM_COL = "is_control_like"

# ============================================================
# END USER CONFIG
# ============================================================


# ----------------------------- Utilities -----------------------------

def load_common_genes(path: str | None) -> list[str] | None:
    """Load a text file of common genes if provided; else return None."""
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] common-genes file not found: {path}")

    genes = [g.strip() for g in path.read_text().splitlines() if g.strip()]
    if len(genes) == 0:
        raise ValueError("[ERROR] common-genes file is empty.")
    
    print(f"[INFO] Loaded {len(genes):,} common genes from {path}")
    return genes


def subset_to_genes(adata: ad.AnnData, genes: list[str], label: str) -> ad.AnnData:
    """Subset an AnnData to given genes, ensuring intersection exists."""
    if genes is None:
        return adata

    present = [g for g in genes if g in adata.var_names]

    print(
        f"[INFO] [{label}] Found {len(present):,} / {len(genes):,} common genes "
        f"({len(genes) - len(present):,} missing)"
    )

    if len(present) == 0:
        raise RuntimeError(
            f"[ERROR] After common-gene filtering, 0 valid genes remain for {label}."
        )

    return adata[:, present].copy()


# ----------------------------- Perturbation Dictionary -----------------------------

def build_test_perturbation_list(common_genes: list[str] | None) -> list[str]:
    """Load dictionary AnnDatas → extract unique perts → optionally subset genes."""
    all_perts = []

    for p in PERT_DICT_PATHS:
        path = Path(p)
        print(f"[INFO] Loading perturbation dictionary: {path}")
        adict = ad.read_h5ad(path)

        # If common-genes specified, subset
        adict = subset_to_genes(adict, common_genes, label=f"dict:{path.name}")

        if DICT_PERT_COL not in adict.obs.columns:
            raise KeyError(
                f"[ERROR] Column '{DICT_PERT_COL}' not found in {path}. "
                f"Update DICT_PERT_COL or fix the dictionary file."
            )

        series = adict.obs[DICT_PERT_COL].astype(str)
        all_perts.append(series)

    if not all_perts:
        raise RuntimeError("[ERROR] No dictionary datasets available.")

    perts = pd.concat(all_perts, ignore_index=True)
    print(f"[INFO] Raw dictionary perturbations: {len(perts):,}")

    # Clean
    perts = perts[~perts.isna()]
    perts = perts[~perts.isin(EXCLUDE_PERTS)]

    unique = perts.drop_duplicates().tolist()
    print(f"[INFO] Unique test perts after filtering: {len(unique):,}")

    return unique


# ----------------------------- Assign Test Perturbations -----------------------------

def assign_test_perts_with_control_like(
    adata: ad.AnnData,
    test_perts: list[str],
    colname: str,
    control_mask_col: str | None,
) -> None:
    """Assign test perturbations to cells, respecting CONTROL_STIM_LABEL."""
    n_cells = adata.n_obs

    values = np.empty(n_cells, dtype=object)

    # Determine which cells are control-like
    if control_mask_col and control_mask_col in adata.obs:
        mask = adata.obs[control_mask_col].astype(bool).values
    else:
        mask = adata.obs[INPUT_PERT_COL].astype(str).values == CONTROL_STIM_LABEL

    n_control = int(mask.sum())
    print(f"[INFO] Control-like '{CONTROL_STIM_LABEL}' rows: {n_control:,} / {n_cells:,}")

    values[mask] = CONTROL_STIM_LABEL

    # Non-control cells
    n_non = n_cells - n_control
    if n_non > 0:
        repeated = (test_perts * ((n_non // len(test_perts)) + 1))[:n_non]
        values[~mask] = repeated

    adata.obs[colname] = values


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build a template AnnData for STATE inference."
    )
    ap.add_argument("--input", required=True, help="Input AnnData (.h5ad)")
    ap.add_argument("--output", required=True, help="Output template .h5ad")
    ap.add_argument(
        "--common-genes",
        required=False,
        help="Optional: Path to common_genes.txt to subset all datasets",
    )
    args = ap.parse_args()

    common_genes = load_common_genes(args.common_genes)

    # Load input dataset
    print(f"[INFO] Loading cytokine AnnData: {args.input}")
    adata = ad.read_h5ad(args.input)

    # Subset to common genes if provided
    adata = subset_to_genes(adata, common_genes, label="cytokine")

    # Ensure required obs columns exist
    for col in (INPUT_CELLTYPE_COL, INPUT_PERT_COL):
        if col not in adata.obs:
            raise KeyError(f"[ERROR] Missing obs column '{col}' in input AnnData.")

    # Subset to target cell type
    mask = adata.obs[INPUT_CELLTYPE_COL] == TARGET_CELLTYPE
    if mask.sum() == 0:
        raise RuntimeError(
            f"[ERROR] No cells of type '{TARGET_CELLTYPE}' found in '{INPUT_CELLTYPE_COL}'."
        )

    adata_ct = adata[mask].copy()
    print(f"[INFO] Selected {adata_ct.n_obs:,} cells × {adata_ct.n_vars:,} genes.")

    # Mark control-like rows
    if MARK_CONTROL_STIM_COL:
        adata_ct.obs[MARK_CONTROL_STIM_COL] = (
            adata_ct.obs[INPUT_PERT_COL].astype(str) == CONTROL_STIM_LABEL
        )
        print(f"[INFO] Marked control-like cells in '{MARK_CONTROL_STIM_COL}'.")

    # Build perturbation list
    test_perts = build_test_perturbation_list(common_genes)

    # Assign test perturbations
    assign_test_perts_with_control_like(
        adata_ct,
        test_perts,
        colname=TEST_PERT_COL,
        control_mask_col=MARK_CONTROL_STIM_COL,
    )

    # Show preview
    print(adata_ct.obs[[INPUT_PERT_COL, TEST_PERT_COL]].head())

    # Save template
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing template AnnData → {out}")
    adata_ct.write_h5ad(out, compression="gzip")

    print("[OK] Template generation complete.")


if __name__ == "__main__":
    main()
