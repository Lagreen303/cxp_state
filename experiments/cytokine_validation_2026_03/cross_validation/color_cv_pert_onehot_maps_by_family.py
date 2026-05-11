#!/usr/bin/env python3
"""
Color the cross-validation perturbation embeddings by cytokine_family.

For each CV fold run directory:
  scFMs/state/experiments/cytokine_validation_2026/cross_validation/run_2026_02_05_fold{K}
we load its `pert_onehot_map.pt` and create a UMAP plot colored by cytokine_family
using the reference AnnData:

  /iridisfs/ddnb/Luke/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings_donors_2.h5ad

Outputs are written to a folder OUTSIDE the run dirs:

  scFMs/state/experiments/cytokine_validation_2026/cross_validation/pert_umaps_by_family/

with files named:

  fold{K}_pert_onehot_map_umap_by_family.png

Run from the project root, for example:

  uv run --with 'anndata,torch,umap-learn,matplotlib,numpy,pandas,adjustText' \\
    python scFMs/state/experiments/cytokine_validation_2026/cross_validation/color_cv_pert_onehot_maps_by_family.py
"""

from collections import Counter
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap


def load_pert_embeddings(pert_path: Path):
    """Load perturbation embeddings dict and stack into matrix + names list."""
    from torch.serialization import add_safe_globals
    import numpy._core.multiarray

    # Allow numpy scalar objects serialized in the checkpoint
    add_safe_globals([numpy._core.multiarray.scalar])

    obj = torch.load(pert_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {pert_path}, got {type(obj)}")

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

    fam_map = (
        df.groupby(name_col)["cytokine_family"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    return fam_map


def make_umap_plot(pert_names, embs, fam_map, out_path: Path):
    """Generate and save a UMAP colored by cytokine_family."""
    families = [fam_map.get(name, "Unknown") for name in pert_names]
    counts = Counter(families)
    print(f"cytokine_family counts for {out_path.name}:", flush=True)
    for fam, n in counts.items():
        print(f"  {fam}: {n}", flush=True)

    reducer = umap.UMAP(random_state=0)
    coords = reducer.fit_transform(embs)

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
    plt.title(f"CV perturbation embeddings colored by cytokine_family ({out_path.stem})")
    plt.legend(
        title="cytokine_family",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        prop={"size": 7},
    )
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}", flush=True)


def main() -> None:
    cv_dir = Path(__file__).resolve().parent
    out_dir = cv_dir / "pert_umaps_by_family"
    ref_adata_path = Path(
        "/iridisfs/ddnb/Luke/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings_donors_2.h5ad"
    )

    print(f"Loading cytokine_family mapping from {ref_adata_path} ...", flush=True)
    fam_map = load_cytokine_family_map(ref_adata_path)
    print(f"Got family annotations for {len(fam_map)} cytokine names", flush=True)

    # Loop over folds 1..9 (adjust if you add more)
    for k in range(1, 10):
        run_dir = cv_dir / f"run_2026_02_05_fold{k}"
        pert_path = run_dir / "pert_onehot_map.pt"
        if not pert_path.exists():
            print(f"[WARN] {pert_path} not found, skipping.", flush=True)
            continue

        print(f"Processing fold {k} from {pert_path} ...", flush=True)
        pert_names, embs = load_pert_embeddings(pert_path)
        print(f"  Loaded {len(pert_names)} perturbations with dim {embs.shape[1]}", flush=True)

        out_png = out_dir / f"fold{k}_pert_onehot_map_umap_by_family.png"
        make_umap_plot(pert_names, embs, fam_map, out_png)

    print("Done generating cross-validation perturbation UMAPs.", flush=True)


if __name__ == "__main__":
    main()

