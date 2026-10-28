import anndata as ad
import scanpy as sc
import logging

adata_path = "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/Parse_10M_PBMC_cytokines.h5ad"
output_path = "/iridisfs/ddnb/Luke/data/parse-10M-90cytokine/parse-10M-90cytokines_processed_state.h5ad"
num_hvgs = 2000

logger = logging.getLogger(__name__)

logger.info(f"Loading AnnData from {adata_path}")
adata = ad.read_h5ad(adata_path)

logger.info("Normalizing total counts per cell")
sc.pp.normalize_total(adata)

logger.info("Applying log1p transformation")
sc.pp.log1p(adata)

logger.info(f"Finding top {num_hvgs} highly variable genes")
sc.pp.highly_variable_genes(adata, n_top_genes=num_hvgs)

logger.info("Storing highly variable genes in .obsm['X_hvg']")
adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()

logger.info(f"Saving preprocessed data to {output_path}")
adata.write_h5ad(output_path)

logger.info(f"Preprocessing complete. Selected {adata.var.highly_variable.sum()} highly variable genes.")