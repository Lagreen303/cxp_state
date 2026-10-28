import anndata as ad
from pathlib import Path

"""
Extract ordered gene list (var_names) from an .h5ad and save as a text file.

This is useful for aligning gene spaces across datasets.

Current defaults:
- Source h5ad: /iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad
- Output txt:  /iridisfs/ddnb/Luke/data/parse-10M-90cytokines/genes_from_adata_for_cellflow_datasets_with_embeddings.txt

Run from anywhere:
  uv run --with anndata python /iridisfs/ddnb/Luke/data/wrangling/extract_gene_list_from_h5ad.py
"""


def main():
    src = Path(
        "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad"
    )
    out_dir = Path("/iridisfs/ddnb/Luke/data/parse-10M-90cytokines")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "genes_from_adata_for_cellflow_datasets_with_embeddings.txt"

    print(f"Reading: {src}", flush=True)
    adata = ad.read_h5ad(src)
    print(f"Shape: {adata.shape}", flush=True)

    genes = adata.var_names.astype(str).tolist()
    print(f"Number of genes: {len(genes)}", flush=True)

    with out_path.open("w") as f:
        for g in genes:
            f.write(f"{g}\n")

    print(f"Wrote gene list to {out_path}", flush=True)


if __name__ == "__main__":
    main()

