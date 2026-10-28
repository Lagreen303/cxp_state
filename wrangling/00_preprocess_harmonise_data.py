import os

# ------------------------------------------------------------
# USER CONFIG — EDIT ONLY THIS BLOCK
# ------------------------------------------------------------

# Input/output paths
INP = "/iridisfs/ddnb/Luke/data/cytokine_replogle_processed/adata_for_cellflow_datasets_with_embeddings_ensembl_state_preprocessed.h5ad"
OUT = "/iridisfs/ddnb/Luke/data/cytokine_replogle_processed/adata_for_cellflow_datasets_with_embeddings_ensembl_state_preprocessed1.h5ad"

# HVGs for STATE manual preprocessing
NUM_HVGS = 2000

# Raw .obs column names in your original dataset
PERT_COL_IN_DATA      = "pert_gene"
BATCH_COL_IN_DATA     = "batch"
CELLTYPE_COL_IN_DATA  = "cell_type"
CONTROL_LABEL_IN_DATA = "non-targeting"

# Standardised names you want
PERT_COL_RENAMED      = "pert_gene"
BATCH_COL_RENAMED     = "batch"
CELLTYPE_COL_RENAMED  = "cell_type"
CONTROL_LABEL_RENAMED = "control"

# ------------------------------------------------------------
# END USER CONFIG
# ------------------------------------------------------------

# Numba cache (HPC/Apptainer safety)
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True, mode=0o755)

import anndata as ad
import numpy as np
import scipy.sparse as sp
import scanpy as sc

print(f"[INFO] Loading: {INP}", flush=True)
adata = ad.read_h5ad(INP)

# ------------------------------------------------------------
# 1) HARD RENAME REQUIRED COLUMNS (no conditionals)
# ------------------------------------------------------------

required_cols = [PERT_COL_IN_DATA, BATCH_COL_IN_DATA, CELLTYPE_COL_IN_DATA]
missing = [c for c in required_cols if c not in adata.obs.columns]
if missing:
    raise KeyError(f"[ERROR] Missing required obs columns: {missing}")

adata.obs.rename(columns={
    PERT_COL_IN_DATA: PERT_COL_RENAMED,
    BATCH_COL_IN_DATA: BATCH_COL_RENAMED,
    CELLTYPE_COL_IN_DATA: CELLTYPE_COL_RENAMED
}, inplace=True)

# remap control label
adata.obs[PERT_COL_RENAMED] = (
    adata.obs[PERT_COL_RENAMED]
    .astype(str)
    .replace({CONTROL_LABEL_IN_DATA: CONTROL_LABEL_RENAMED})
)

print(f"[INFO] Renamed {PERT_COL_IN_DATA} → {PERT_COL_RENAMED}")
print(f"[INFO] Renamed {BATCH_COL_IN_DATA} → {BATCH_COL_RENAMED}")
print(f"[INFO] Renamed {CELLTYPE_COL_IN_DATA} → {CELLTYPE_COL_RENAMED}")
print(f"[INFO] Control '{CONTROL_LABEL_IN_DATA}' → '{CONTROL_LABEL_RENAMED}'", flush=True)

# ------------------------------------------------------------
# 2) STATE COUNTS FIX
# ------------------------------------------------------------

# print("[INFO] Ensuring counts are STATE-compatible...", flush=True)

# # Prefer layers["counts"], else X is assumed to be counts
# if "counts" in adata.layers:
#     counts = adata.layers["counts"]
#     print("[INFO] Using layers['counts'] as counts", flush=True)
# else:
#     counts = adata.X
#     print("[INFO] Using X as counts", flush=True)

# # convert to CSR float32 + clip negatives
# if sp.issparse(counts):
#     counts = counts.tocsr().astype(np.float32, copy=False)
#     counts.data[counts.data < 0] = 0
# else:
#     counts = np.asarray(counts, dtype=np.float32)
#     counts[counts < 0] = 0
#     counts = sp.csr_matrix(counts)

# # drop all-zero genes
# gene_sums = np.array(counts.sum(axis=0)).ravel()
# keep = gene_sums > 0
# dropped = (~keep).sum()
# if dropped:
#     print(f"[INFO] Dropping {dropped} all-zero genes", flush=True)
#     adata = adata[:, keep].copy()
#     counts = counts[:, keep]

# # write cleaned counts into layers and X
# adata.layers["counts"] = counts
# adata.X = counts

# print(f"[INFO] Counts ready: {adata.n_obs} cells × {adata.n_vars} genes", flush=True)

# ------------------------------------------------------------
# 3) STATE MANUAL PREPROCESSING (exact official steps)
# ------------------------------------------------------------

# print("[INFO] Normalizing total counts per cell...")
# sc.pp.normalize_total(adata)

# print("[INFO] Applying log1p...")
# sc.pp.log1p(adata)

# print(f"[INFO] Selecting top {NUM_HVGS} HVGs...")
# sc.pp.highly_variable_genes(adata, n_top_genes=NUM_HVGS)

# # store HVG matrix in obsm["X_hvg"]
# hvg_mask = adata.var["highly_variable"].values
# print("[INFO] Storing HVGs in obsm['X_hvg']...")
# adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()

# ------------------------------------------------------------
# 4) WRITE OUTPUT
# ------------------------------------------------------------

print(f"[INFO] Saving preprocessed AnnData → {OUT}", flush=True)
adata.write_h5ad(OUT, compression="gzip")

print("[OK] Done.", flush=True)
