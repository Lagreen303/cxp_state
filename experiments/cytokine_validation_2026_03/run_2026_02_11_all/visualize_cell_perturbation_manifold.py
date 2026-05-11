#!/usr/bin/env python3
"""
Visualize the cell/perturbation manifold from a STATE checkpoint.

Extracts basal + perturbation embeddings from internal (training) and external validation
data to understand poor external generalization (strong internal, weak external).

Inputs
------
- Model: run_2026_02_11_all/checkpoints/step=80000.ckpt
- Training data: Ahmed parse-10M-90cytokines (via cytokine.toml)
- External data:
  - GSE96583 IFNbeta: aligned_avg (genes imputed)
  - GSE178429 IFN-y: aligned_avg (genes imputed)

Outputs
-------
- UMAP plots colored by: dataset, cytokine, cell_type
- Saved to run_2026_02_11_all/cell_perturbation_manifold_*.png
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scanpy as sc
import torch
import umap
import yaml
from tqdm import tqdm

# Add state src to path
SCRIPTS_DIR = Path(__file__).resolve().parent
STATE_ROOT = SCRIPTS_DIR.parent.parent.parent
sys.path.insert(0, str(STATE_ROOT / "src"))

# Default paths
DEFAULT_RUN_DIR = Path(
    "/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/run_2026_02_11_all"
)
DEFAULT_TRAIN_ADATA = "/iridisfs/ddnb/Ahmed/data/parse-10M-90cytokines/adata_for_cellflow_datasets_with_embeddings.h5ad"
DEFAULT_EXTERNAL_IFNBETA = "/iridisfs/ddnb/Luke/data/GSE96583_Kang/GSE96583_batch2_IFNbeta_aligned_avg.h5ad"
DEFAULT_EXTERNAL_IFNY = "/iridisfs/ddnb/Luke/data/GSE178429_IFN-y_PBMCs/GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg.h5ad"
# Preprocessed (gene-aligned) templates for inference
DEFAULT_TEMPLATE_IFNBETA = (
    "/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/templates/"
    "GSE96583_batch2_IFNbeta_aligned_avg_processed.h5ad"
)
DEFAULT_TEMPLATE_IFNY = (
    "/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/templates/"
    "GSE178429_PBMCs_IFN_Control_1h6h_aligned_avg_processed.h5ad"
)


def to_dense(mat):
    """Return dense numpy array for AnnData .X."""
    try:
        import scipy.sparse as sp
        if sp.issparse(mat):
            return mat.toarray()
    except Exception:
        pass
    return np.asarray(mat)


def load_config_model(run_dir: Path, checkpoint: str):
    """Load config, model, and mappings."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load data module for var_dims and pert_onehot_map
    data_module_path = run_dir / "data_module.torch"
    if not data_module_path.exists():
        raise FileNotFoundError(f"Data module not found: {data_module_path}")

    from cell_load.data_modules import PerturbationDataModule
    dm = PerturbationDataModule.load_state(str(data_module_path))
    dm.setup(stage="test")

    var_dims = dm.get_var_dims()
    pert_onehot_map = dm.pert_onehot_map
    control_pert = dm.get_control_pert()

    # Load model
    ckpt_path = run_dir / "checkpoints" / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_kwargs = cfg["model"]["kwargs"]
    model_init = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }
    model = StateTransitionPerturbationModel.load_from_checkpoint(
        str(ckpt_path), weights_only=False, **model_init
    )
    model.eval()
    device = next(model.parameters()).device

    cell_set_len = getattr(model, "cell_sentence_len", 128)
    return cfg, model, dm, var_dims, pert_onehot_map, control_pert, device, cell_set_len


def extract_embeddings_from_batch(
    model, batch: dict, device, cell_sentence_len: int
) -> np.ndarray:
    """
    Extract combined (basal + pert) embeddings from a batch.
    Batch has ctrl_cell_emb [N, input_dim], pert_emb [N, pert_dim].
    Reshape to (B, S, dim) and run encoders. Returns [N, hidden_dim].
    """
    ctrl = batch["ctrl_cell_emb"].to(device)
    pert = batch["pert_emb"].to(device)
    N = ctrl.shape[0]
    if ctrl.ndim == 2:
        S = cell_sentence_len
        B = (N + S - 1) // S
        pad = B * S - N
        if pad > 0:
            ctrl = torch.nn.functional.pad(ctrl, (0, 0, 0, pad), mode="replicate")
            pert = torch.nn.functional.pad(pert, (0, 0, 0, pad), mode="replicate")
        ctrl = ctrl.reshape(B, S, -1)
        pert = pert.reshape(B, S, -1)
    pert_emb = model.encode_perturbation(pert)
    basal_emb = model.encode_basal_expression(ctrl)
    combined = pert_emb + basal_emb
    out = combined.detach().cpu().numpy().reshape(-1, combined.shape[-1])
    return out[:N]


def collect_internal_embeddings(
    model,
    dm,
    device,
    cell_set_len: int,
    max_cells: int = 50000,
) -> tuple:
    """Collect embeddings from training data."""
    loader = dm.train_dataloader(test=True)
    if loader.batch_sampler is not None:
        tot = getattr(loader.batch_sampler, "tot_num", None) or len(loader.dataset)
    else:
        tot = len(loader.dataset)
    max_cells = min(max_cells, tot)

    embs_list = []
    perts_list = []
    celltypes_list = []
    n = 0
    pbar = tqdm(loader, desc="Internal (train)", total=(max_cells // (dm.batch_size * cell_set_len)) + 1)
    with torch.no_grad():
        for batch in pbar:
            if n >= max_cells:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            emb = extract_embeddings_from_batch(model, batch, device, cell_set_len)
            embs_list.append(emb)
            pert_names = batch.get("pert_name", [])
            cts = batch.get("cell_type", [])
            n_emb = emb.shape[0]
            if isinstance(pert_names, list) and len(pert_names) >= n_emb:
                perts_list.extend(pert_names[:n_emb])
            else:
                perts_list.extend([str(pert_names[0]) if pert_names else "unknown"] * n_emb)
            if isinstance(cts, list) and len(cts) >= n_emb:
                celltypes_list.extend(cts[:n_emb])
            else:
                celltypes_list.extend([str(cts[0]) if cts else "unknown"] * n_emb)
            n += emb.shape[0]
            pbar.set_postfix(n=n)

    embs = np.vstack(embs_list)[:max_cells]
    perts = np.array(perts_list[:max_cells])
    celltypes = np.array(celltypes_list[:max_cells])
    return embs, perts, celltypes


def collect_external_embeddings(
    model,
    adata_path: str,
    pert_onehot_map: dict,
    control_pert: str,
    pert_col: str,
    cell_type_col: str,
    device,
    cell_set_len: int,
    dataset_label: str,
    default_pert_vec: torch.Tensor,
) -> tuple:
    """Collect embeddings from external AnnData (processed, gene-aligned)."""
    adata = sc.read_h5ad(adata_path)
    X = to_dense(adata.X).astype(np.float32)
    pert_names = adata.obs[pert_col].astype(str).values
    if cell_type_col and cell_type_col in adata.obs:
        celltypes = adata.obs[cell_type_col].astype(str).values
    else:
        celltypes = np.array(["unknown"] * adata.n_obs)

    ctl_mask = pert_names == str(control_pert)
    ctrl_pool = np.where(ctl_mask)[0]
    pert_idx = np.where(~ctl_mask)[0]
    if len(ctrl_pool) == 0:
        raise ValueError(f"No control cells for '{control_pert}' in {adata_path}")

    rng = np.random.RandomState(42)
    embs_list = []
    perts_list = []
    celltypes_list = []

    for start in tqdm(range(0, len(pert_idx), cell_set_len), desc=f"External ({dataset_label})"):
        end = min(start + cell_set_len, len(pert_idx))
        idx_window = pert_idx[start:end]
        win_size = len(idx_window)

        sampled_ctrl = rng.choice(ctrl_pool, size=win_size, replace=True)
        ctrl_basal = X[sampled_ctrl, :]
        pert_names_win = pert_names[idx_window]

        pert_vecs = []
        for p in pert_names_win:
            v = pert_onehot_map.get(p, default_pert_vec)
            if torch.is_tensor(v):
                v = v.cpu().numpy()
            else:
                v = np.asarray(v)
            pert_vecs.append(v)
        pert_oh = np.stack(pert_vecs, axis=0).astype(np.float32)

        ctrl_t = torch.tensor(ctrl_basal, dtype=torch.float32, device=device)
        pert_t = torch.tensor(pert_oh, dtype=torch.float32, device=device)
        batch = {
            "ctrl_cell_emb": ctrl_t.unsqueeze(0),
            "pert_emb": pert_t.unsqueeze(0),
        }
        with torch.no_grad():
            emb = extract_embeddings_from_batch(model, batch, device, cell_set_len)
        embs_list.append(emb)
        perts_list.extend(pert_names_win.tolist())
        celltypes_list.extend(celltypes[idx_window].tolist())

    embs = np.vstack(embs_list)
    perts = np.array(perts_list)
    celltypes = np.array(celltypes_list)
    return embs, perts, celltypes


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    alpha: float = 0.6,
):
    """Plot UMAP colored by labels."""
    import matplotlib.pyplot as plt

    unique = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique))))
    if len(unique) > 20:
        colors = plt.cm.turbo(np.linspace(0, 1, len(unique)))
    color_map = {u: colors[i] for i, u in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for u in unique:
        mask = labels == u
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=alpha,
            c=[color_map[u]],
            label=u,
            edgecolors="none",
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize cell/perturbation manifold")
    ap.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--checkpoint", type=str, default="step=80000.ckpt")
    ap.add_argument("--max-internal", type=int, default=30000)
    ap.add_argument("--include-ifnbeta", action="store_true", default=True)
    ap.add_argument("--include-ifny", action="store_true", default=True)
    ap.add_argument("--template-ifnbeta", type=str, default=DEFAULT_TEMPLATE_IFNBETA)
    ap.add_argument("--template-ifny", type=str, default=DEFAULT_TEMPLATE_IFNY)
    ap.add_argument("--external-control-pert", type=str, default="control")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir or run_dir)

    print("Loading config and model...")
    cfg, model, dm, var_dims, pert_onehot_map, control_pert, device, cell_set_len = load_config_model(
        run_dir, args.checkpoint
    )
    data_cfg = cfg.get("data", {}).get("kwargs", {})
    pert_col = data_cfg.get("pert_col", "cytokine")
    cell_type_col = data_cfg.get("cell_type_key", "cell_type")

    pert_dim = var_dims["pert_dim"]
    default_pert_vec = pert_onehot_map.get(control_pert)
    if default_pert_vec is None:
        default_pert_vec = torch.zeros(pert_dim, dtype=torch.float32)
        if pert_dim > 0:
            default_pert_vec[0] = 1.0
    elif torch.is_tensor(default_pert_vec):
        default_pert_vec = default_pert_vec.clone()
    else:
        default_pert_vec = torch.tensor(default_pert_vec, dtype=torch.float32)

    # 1) Internal embeddings
    print("Collecting internal (training) embeddings...")
    embs_int, perts_int, cts_int = collect_internal_embeddings(
        model, dm, device, cell_set_len, max_cells=args.max_internal
    )
    dataset_int = np.array(["internal"] * len(embs_int))

    all_embs = [embs_int]
    all_perts = [perts_int]
    all_cts = [cts_int]
    all_datasets = [dataset_int]

    # 2) External GSE96583 IFNbeta
    ext_control = getattr(args, "external_control_pert", "control")
    if args.include_ifnbeta and os.path.exists(args.template_ifnbeta):
        print("Collecting external GSE96583 IFNbeta embeddings...")
        embs_ext, perts_ext, cts_ext = collect_external_embeddings(
            model,
            args.template_ifnbeta,
            pert_onehot_map,
            ext_control,
            pert_col,
            cell_type_col,
            device,
            cell_set_len,
            "GSE96583_IFNbeta",
            default_pert_vec,
        )
        all_embs.append(embs_ext)
        all_perts.append(perts_ext)
        all_cts.append(cts_ext)
        all_datasets.append(np.array(["GSE96583_IFNbeta"] * len(embs_ext)))

    # 3) External GSE178429 IFN-y
    if args.include_ifny and os.path.exists(args.template_ifny):
        print("Collecting external GSE178429 IFN-y embeddings...")
        embs_ext2, perts_ext2, cts_ext2 = collect_external_embeddings(
            model,
            args.template_ifny,
            pert_onehot_map,
            ext_control,
            pert_col,
            cell_type_col,
            device,
            cell_set_len,
            "GSE178429_IFNy",
            default_pert_vec,
        )
        all_embs.append(embs_ext2)
        all_perts.append(perts_ext2)
        all_cts.append(cts_ext2)
        all_datasets.append(np.array(["GSE178429_IFNy"] * len(embs_ext2)))

    embs = np.vstack(all_embs)
    perts = np.concatenate(all_perts)
    celltypes = np.concatenate(all_cts)
    datasets = np.concatenate(all_datasets)

    print(f"Total embeddings: {embs.shape[0]}")

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embs)

    # Plots
    plot_umap(coords, datasets, "Cell/perturbation manifold: dataset", out_dir / "manifold_by_dataset.png")
    plot_umap(coords, perts, "Cell/perturbation manifold: cytokine", out_dir / "manifold_by_cytokine.png")
    plot_umap(coords, celltypes, "Cell/perturbation manifold: cell type", out_dir / "manifold_by_celltype.png")

    # Save coords and labels for downstream use
    np.save(out_dir / "manifold_coords.npy", coords)
    np.save(out_dir / "manifold_datasets.npy", datasets)
    np.save(out_dir / "manifold_perts.npy", perts)
    np.save(out_dir / "manifold_celltypes.npy", celltypes)

    print(f"Done. Plots and arrays saved to {out_dir}")


if __name__ == "__main__":
    main()
