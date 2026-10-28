#!/usr/bin/env python3
"""Merge Collectri train1+test1 X_cxp h5ads and split by VAE train2/test2 donors.

Produces all-cytokine donor-split files for STATE tx (no cytokine holdout).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import anndata as ad

DEFAULT_TRAIN1 = Path(
    "/iridisfs/ddnb/Luke/cellxpathway/outputs/v2/v2_collectri/h5ad/x_cxp_train1.h5ad"
)
DEFAULT_TEST1 = Path(
    "/iridisfs/ddnb/Luke/cellxpathway/outputs/v2/v2_collectri/h5ad/test1_x_cxp.h5ad"
)
DEFAULT_OUT = Path(
    "/iridisfs/ddnb/Luke/cellxpathway/outputs/v2/v2_collectri/h5ad_allcyto_donor"
)
DEFAULT_SPLIT = Path("/iridisfs/ddnb/Luke/cellxpathway/configs/v2/train2_test2.json")
DEFAULT_EMBED_KEY = "X_cxp"
DEFAULT_TRAIN2_NAME = "x_cxp_allcyto_train2.h5ad"
DEFAULT_TEST2_NAME = "x_cxp_allcyto_test2.h5ad"


def _load_donor_lists(split_json: Path) -> tuple[set[str], set[str]]:
    cfg = json.loads(split_json.read_text())
    train = {str(d) for d in cfg["train2_donors"]}
    test = {str(d) for d in cfg["test2_donors"]}
    if train & test:
        raise ValueError(f"train2/test2 donor overlap: {sorted(train & test)}")
    return train, test


def _subset_by_donors(adata: ad.AnnData, donors: set[str]) -> ad.AnnData:
    mask = adata.obs["donor"].astype(str).isin(donors).to_numpy()
    if not bool(mask.any()):
        raise ValueError(f"no cells for donors {sorted(donors)}")
    return adata[mask].copy()


def _donors_in_file(path: Path) -> set[str]:
    backed = ad.read_h5ad(path, backed="r")
    if "donor" not in backed.obs.columns:
        backed.file.close()
        raise KeyError(f"{path} missing obs.donor")
    donors = set(backed.obs["donor"].astype(str).unique())
    backed.file.close()
    return donors


def _subset_by_donors_from_path(path: Path, donors: set[str], embed_key: str, label: str) -> ad.AnnData:
    """Backed read; materialize only the donor subset (avoids full train1 in RAM)."""
    print(f"[INFO] reading {label} (backed): {path}", flush=True)
    backed = ad.read_h5ad(path, backed="r")
    if "donor" not in backed.obs.columns:
        backed.file.close()
        raise KeyError(f"{label} missing obs.donor")
    if embed_key not in backed.obsm:
        backed.file.close()
        raise KeyError(f"{label} missing obsm['{embed_key}']")
    print(f"[INFO] {label} shape={backed.shape}", flush=True)
    mask = backed.obs["donor"].astype(str).isin(donors).to_numpy()
    if not bool(mask.any()):
        backed.file.close()
        raise ValueError(f"no cells for donors {sorted(donors)} in {label}")
    out = backed[mask].to_memory()
    backed.file.close()
    gc.collect()
    return out


def _summarize(adata: ad.AnnData, embed_key: str) -> dict:
    donors = sorted(adata.obs["donor"].astype(str).unique().tolist())
    cyts = sorted(adata.obs["cytokine"].astype(str).unique().tolist())
    return {
        "n_cells": int(adata.n_obs),
        "n_var": int(adata.n_vars),
        "n_cytokines": len(cyts),
        "donors": donors,
        "cytokines": cyts,
        "embed_key": embed_key,
        f"has_{embed_key}": embed_key in adata.obsm,
        f"{embed_key}_shape": list(adata.obsm[embed_key].shape) if embed_key in adata.obsm else None,
    }


def _concat_parts(parts: list[ad.AnnData], embed_key: str) -> ad.AnnData:
    out = ad.concat(parts, axis=0, join="inner", merge="same", index_unique=None)
    if out.obs_names.has_duplicates:
        n_dup = int(out.obs_names.duplicated().sum())
        print(f"[WARN] dropping {n_dup} duplicate barcodes", flush=True)
        out = out[~out.obs_names.duplicated()].copy()
    if embed_key not in out.obsm:
        raise KeyError(f"concat result missing obsm['{embed_key}']")
    return out


def build(
    *,
    train1: Path,
    test1: Path,
    out_dir: Path,
    split_json: Path,
    embed_key: str = DEFAULT_EMBED_KEY,
    train2_name: str = DEFAULT_TRAIN2_NAME,
    test2_name: str = DEFAULT_TEST2_NAME,
) -> dict:
    train_donors, test_donors = _load_donor_lists(split_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / train2_name
    test_out = out_dir / test2_name

    train_parts: list[ad.AnnData] = []
    test_parts: list[ad.AnnData] = []
    var_names_ref: list[str] | None = None
    seen_donors: set[str] = set()

    for label, path in (("train1", train1), ("test1", test1)):
        seen_donors |= _donors_in_file(path)
        backed = ad.read_h5ad(path, backed="r")
        names = list(backed.var_names)
        if var_names_ref is None:
            var_names_ref = names
        elif names != var_names_ref:
            backed.file.close()
            raise ValueError(f"{label} var_names mismatch vs train1")
        backed.file.close()
        train_parts.append(_subset_by_donors_from_path(path, train_donors, embed_key, label))
        test_parts.append(_subset_by_donors_from_path(path, test_donors, embed_key, label))

    expected = train_donors | test_donors
    if seen_donors != expected:
        raise ValueError(
            f"donor set mismatch: extra={sorted(seen_donors - expected)} "
            f"missing={sorted(expected - seen_donors)}"
        )

    print(f"[INFO] writing train donors {sorted(train_donors)} → {train_out}", flush=True)
    train_ad = _concat_parts(train_parts, embed_key)
    del train_parts
    gc.collect()
    train_summary = _summarize(train_ad, embed_key)
    train_ad.write_h5ad(train_out)
    del train_ad
    gc.collect()

    print(f"[INFO] writing test donors {sorted(test_donors)} → {test_out}", flush=True)
    test_ad = _concat_parts(test_parts, embed_key)
    del test_parts
    gc.collect()
    test_summary = _summarize(test_ad, embed_key)
    test_ad.write_h5ad(test_out)
    del test_ad
    gc.collect()

    counts = {
        "split_id": json.loads(split_json.read_text()).get("split_id", "vae_train2_test2_v1"),
        "train2_donors": sorted(train_donors),
        "test2_donors": sorted(test_donors),
        "inputs": {"train1": str(train1), "test1": str(test1)},
        "outputs": {"train2": str(train_out), "test2": str(test_out)},
        "train2": train_summary,
        "test2": test_summary,
    }
    train_cyts = set(train_summary["cytokines"])
    test_cyts = set(test_summary["cytokines"])
    counts["cytokine_set_equal"] = train_cyts == test_cyts
    counts["n_cytokines_union"] = len(train_cyts | test_cyts)
    if not counts["cytokine_set_equal"]:
        counts["cytokines_only_train2"] = sorted(train_cyts - test_cyts)
        counts["cytokines_only_test2"] = sorted(test_cyts - train_cyts)
        print(
            "[WARN] train2/test2 cytokine sets differ "
            f"(only_train={len(train_cyts - test_cyts)} only_test={len(test_cyts - train_cyts)})",
            flush=True,
        )
    (out_dir / "export_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: counts[k]
                for k in ("train2", "test2", "n_cytokines_union", "cytokine_set_equal")
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[OK] wrote {train_out} and {test_out}", flush=True)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train1", type=Path, default=DEFAULT_TRAIN1)
    p.add_argument("--test1", type=Path, default=DEFAULT_TEST1)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--embed-key", type=str, default=DEFAULT_EMBED_KEY)
    p.add_argument("--train2-name", type=str, default=DEFAULT_TRAIN2_NAME)
    p.add_argument("--test2-name", type=str, default=DEFAULT_TEST2_NAME)
    args = p.parse_args(argv)
    for path in (args.train1, args.test1, args.split_json):
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            return 1
    build(
        train1=args.train1.resolve(),
        test1=args.test1.resolve(),
        out_dir=args.out_dir.resolve(),
        split_json=args.split_json.resolve(),
        embed_key=str(args.embed_key),
        train2_name=str(args.train2_name),
        test2_name=str(args.test2_name),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
