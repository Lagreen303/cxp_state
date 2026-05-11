#!/usr/bin/env python3
"""
Color the perturbation embeddings from pert_onehot_map.pt by cytokine_family.

Inputs
------
- Pert embeddings:
    scFMs/state/experiments/cytokine_validation_2026/run_2026_02_11_all/pert_onehot_map.pt
  (dict: perturbation name -> 2560-dim torch.Tensor)

- Reference AnnData (for cytokine_family):
    /iridisfs/ddnb/Luke/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings_donors_2.h5ad

Outputs
-------
- UMAP plot colored by cytokine_family:
    pert_onehot_map_umap_by_family.png
"""

from pathlib import Path
from collections import Counter

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from scipy import sparse


def load_pert_embeddings(pert_path: Path):
    """Load perturbation embeddings dict and stack into matrix + names list."""
    from torch.serialization import add_safe_globals
    import numpy._core.multiarray

    # Allow numpy scalar objects in the checkpoint
    add_safe_globals([numpy._core.multiarray.scalar])

    obj = torch.load(pert_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict from {pert_path}, got {type(obj)}")

    names = list(obj.keys())
    embs = torch.stack([obj[k] for k in names], dim=0).numpy()
    return names, embs


def load_cytokine_family_map(adata_path: Path):
    """Build mapping: cytokine name -> cytokine_family from reference AnnData."""
    adata = ad.read_h5ad(adata_path)
    obs = adata.obs

    # Infer the column holding cytokine / perturbation names
    name_cols = [c for c in obs.columns if "cytokine" in c.lower() or "pert" in c.lower() or "ligand" in c.lower()]
    if "cytokine" in obs.columns:
        name_col = "cytokine"
    elif name_cols:
        name_col = name_cols[0]
    else:
        raise KeyError("Could not find a cytokine/perturbation name column in adata.obs")

    if "cytokine_family" not in obs.columns:
        raise KeyError("'cytokine_family' not found in adata.obs")

    df = obs[[name_col, "cytokine_family"]].copy().dropna(subset=[name_col])

    # Majority vote per cytokine name
    fam_map = (
        df.groupby(name_col)["cytokine_family"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    return fam_map


def main() -> None:
    run_dir = Path(__file__).resolve().parent
    pert_path = run_dir / "pert_onehot_map.pt"
    ref_adata_path = Path(
        "/iridisfs/ddnb/Luke/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings_donors_2.h5ad"
    )
    out_png = run_dir / "pert_onehot_map_umap_by_family.png"

    print(f"Loading perturbation embeddings from {pert_path} ...", flush=True)
    pert_names, embs = load_pert_embeddings(pert_path)
    print(f"Loaded {len(pert_names)} perturbations with embedding dim {embs.shape[1]}", flush=True)

    print(f"Loading cytokine_family mapping from {ref_adata_path} ...", flush=True)
    fam_map = load_cytokine_family_map(ref_adata_path)
    print(f"Got family annotations for {len(fam_map)} cytokine names", flush=True)

    # Map each perturbation to a family
    families = [fam_map.get(name, "Unknown") for name in pert_names]
    counts = Counter(families)
    print("cytokine_family counts:", flush=True)
    for fam, n in counts.items():
        print(f"  {fam}: {n}", flush=True)

    # Compute UMAP
    reducer = umap.UMAP(random_state=0)
    coords = reducer.fit_transform(embs)

    # Color by family
    unique_fams = sorted(counts.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_fams)))
    color_map = {fam: colors[i] for i, fam in enumerate(unique_fams)}

    plt.figure(figsize=(8, 4.8))
    for fam in unique_fams:
        mask = np.array([f == fam for f in families])
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=25,
            alpha=0.9,
            color=color_map[fam],
            label=fam,
            edgecolor="none",
        )

    # Add labels; optionally adjust to reduce overlap
    texts = []
    for x, y, lab in zip(coords[:, 0], coords[:, 1], pert_names):
        texts.append(plt.text(x, y, lab, fontsize=7, alpha=0.9))

    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            expand_points=(1.1, 1.4),
            expand_text=(1.1, 1.4),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
        )
    except Exception as e:
        print(f"adjustText not available or failed ({e}); proceeding without label adjustment.", flush=True)

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("STATE perturbation embeddings colored by cytokine_family")
    plt.legend(
        title="cytokine_family",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        prop={"size": 7},
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved colored UMAP to {out_png}", flush=True)


if __name__ == "__main__":
    main()

