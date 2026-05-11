#!/usr/bin/env python3
"""Add var['gene_name'] (HGNC) from Ensembl var_names using a GENCODE GTF."""
from __future__ import annotations

import argparse
import gzip
import re
import sys


def parse_gtf_ensembl_to_symbol(gtf_path: str) -> dict[str, str]:
    ens_to_symbol: dict[str, str] = {}
    opener = gzip.open if gtf_path.endswith(".gz") else open
    mode = "rt" if gtf_path.endswith(".gz") else "r"
    with opener(gtf_path, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = fields[8]
            gid = re.search(r'gene_id "([^"]+)"', attrs)
            gname = re.search(r'gene_name "([^"]+)"', attrs)
            if gid and gname:
                ens = gid.group(1).split(".")[0]
                ens_to_symbol[ens] = gname.group(1)
    return ens_to_symbol


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input .h5ad")
    p.add_argument("--output", required=True, help="Output .h5ad (new file)")
    p.add_argument(
        "--gtf",
        default="/iridisfs/ddnb/Luke/scFMs/scGPT/gencode.v48.basic.annotation.gtf.gz",
        help="GENCODE GTF.gz with gene_id and gene_name",
    )
    p.add_argument(
        "--column",
        default="gene_name",
        help="Name for the new var column (default gene_name)",
    )
    args = p.parse_args()

    import anndata as ad

    print("Loading GTF map…", flush=True)
    m = parse_gtf_ensembl_to_symbol(args.gtf)
    print(f"  {len(m)} Ensembl→symbol entries", flush=True)

    print(f"Reading {args.input} …", flush=True)
    adata = ad.read_h5ad(args.input)

    clean = adata.var_names.astype(str).str.replace(r"\..*$", "", regex=True)
    adata.var[args.column] = clean.map(lambda x: m.get(x, ""))
    n_ok = (adata.var[args.column].astype(str).str.len() > 0).sum()
    print(f"Mapped {n_ok} / {adata.n_vars} genes to non-empty symbols", flush=True)
    if n_ok == 0:
        print("ERROR: no symbols mapped — check GTF/build matches var_names.", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {args.output} …", flush=True)
    adata.write_h5ad(args.output)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
