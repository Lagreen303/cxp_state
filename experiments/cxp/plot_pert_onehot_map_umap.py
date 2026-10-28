#!/usr/bin/env python3
"""UMAP of STATE pert_onehot_map.pt (ESM2 perturbation features) for a CxP run dir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
import yaml


def _assign_family(name: str, families_cfg: dict) -> str:
    fams = families_cfg.get("families") or {}
    for fam, spec in fams.items():
        if fam == "other":
            continue
        if isinstance(spec, list) and name in spec:
            return fam
        if isinstance(spec, dict) and spec.get("prefix") and str(name).startswith(spec["prefix"]):
            return fam
    return "other"


def _split_role(name: str, split: dict) -> str:
    if name == split.get("control_cytokine", "PBS"):
        return "control"
    if name in split.get("val_cytokines", []):
        return "val"
    if name in split.get("test_cytokines", []):
        return "test"
    return "train"


def load_embeddings(pert_path: Path) -> tuple[list[str], np.ndarray]:
    obj = torch.load(pert_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {pert_path}, got {type(obj)}")
    names = [str(k) for k in obj.keys()]
    embs = torch.stack([obj[k] for k in names], dim=0).float().numpy()
    return names, embs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "CollecTRI" / "run_cxp_v2_collectri",
    )
    ap.add_argument(
        "--split-json",
        type=Path,
        default=Path("/iridisfs/ddnb/Luke/cellxpathway/configs/v2/cytokine_split_80_5_5.json"),
    )
    ap.add_argument(
        "--families-yaml",
        type=Path,
        default=Path("/iridisfs/ddnb/Luke/cellxpathway/configs/pp/cytokine_families.yaml"),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    pert_path = run_dir / "pert_onehot_map.pt"
    if not pert_path.is_file():
        raise FileNotFoundError(pert_path)

    split = json.loads(args.split_json.read_text())
    families_cfg = yaml.safe_load(args.families_yaml.read_text())

    names, embs = load_embeddings(pert_path)
    meta = pd.DataFrame(
        {
            "perturbation": names,
            "split_role": [_split_role(n, split) for n in names],
            "family": [_assign_family(n, families_cfg) for n in names],
        }
    )

    reducer = umap.UMAP(random_state=args.seed, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embs)
    meta["umap1"] = coords[:, 0]
    meta["umap2"] = coords[:, 1]
    meta.to_csv(run_dir / "pert_onehot_map_umap_coords.csv", index=False)

    out_split = run_dir / "figures" / "pert_onehot_map_umap_by_split"
    out_split.mkdir(parents=True, exist_ok=True)
    roles = ["train", "val", "test", "control"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for ax, role in zip(axes.ravel(), roles):
        bg = meta["split_role"] != role
        hi = meta["split_role"] == role
        ax.scatter(meta.loc[bg, "umap1"], meta.loc[bg, "umap2"], s=18, c="#d0d0d0", alpha=0.5, linewidths=0)
        ax.scatter(
            meta.loc[hi, "umap1"],
            meta.loc[hi, "umap2"],
            s=42,
            c="#1f4e79",
            alpha=0.95,
            linewidths=0.3,
            edgecolors="white",
        )
        for _, row in meta.loc[hi].iterrows():
            ax.annotate(
                row["perturbation"],
                (row["umap1"], row["umap2"]),
                fontsize=6,
                ha="center",
                va="bottom",
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_title(f"{role} (n={int(hi.sum())})")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig.suptitle(f"Perturbation ESM2 embeddings — {run_dir.name}", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_split.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # IFN-beta context: same coords, highlight IFN-beta vs other val cytokines
    out_ifn = run_dir / "figures" / "pert_onehot_map_umap_ifn_beta_context"
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    panels = [
        ("IFN-beta", meta["perturbation"] == "IFN-beta"),
        ("other val", meta["split_role"] == "val"),
        ("train", meta["split_role"] == "train"),
    ]
    for ax, (title, mask) in zip(axes2, panels):
        ax.scatter(meta.loc[~mask, "umap1"], meta.loc[~mask, "umap2"], s=16, c="#e0e0e0", alpha=0.45, linewidths=0)
        ax.scatter(
            meta.loc[mask, "umap1"],
            meta.loc[mask, "umap2"],
            s=40,
            c="#b22222" if title == "IFN-beta" else "#1f4e79",
            alpha=0.9,
            linewidths=0,
        )
        for _, row in meta.loc[mask].iterrows():
            ax.annotate(
                row["perturbation"],
                (row["umap1"], row["umap2"]),
                fontsize=6,
                ha="center",
                va="bottom",
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig2.suptitle("IFN-beta among val / train perturbation embeddings", y=1.05)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(out_ifn.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Wrote {out_split.with_suffix('.png')}")
    print(f"Wrote {out_ifn.with_suffix('.png')}")
    print(f"Wrote {run_dir / 'pert_onehot_map_umap_coords.csv'}")


if __name__ == "__main__":
    main()
