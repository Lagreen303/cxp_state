#!/usr/bin/env python3
"""
Generate 1 template .h5ad per test perturbation (e.g. 2000 Replogle genes).

Each output file contains the SAME baseline cells (optionally filtered),
but with:
  - obs[TEST_PERT_COL] set to:
        * CONTROL_LABEL for a subset of cells (controls) IF CONTROL_LABEL is not None
        * <pert> for the remaining cells
  - unique obs_names (barcodes) by appending __<pert>

Intended for STATE inference templates (before state tx preprocess_inference).
Run inference with:
  - if CONTROL_LABEL is not None:
        --pert-col <TEST_PERT_COL> --control-pert <CONTROL_LABEL>
  - if CONTROL_LABEL is None:
        --pert-col <TEST_PERT_COL>   (DO NOT pass --control-pert)

None-safe config logic:
- If COMMON_GENES_PATH is None -> no gene filtering
- If CELLTYPE_KEEP is None -> keep all cell types
- If PERT_COL is None OR CYTOKINE_KEEP is None -> no cytokine/pert filtering
- If CONTROL_LABEL is None -> no controls are created
"""

from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd


# ============================================================
# CONFIG — EDIT THIS BLOCK
# ============================================================

# Baseline pool
INPUT_H5AD = "/iridisfs/ddnb/Luke/data/cytokine_1k1k/eQTLAutoimmune_aligned.h5ad"

# Output folder
OUTPUT_DIR = "/iridisfs/ddnb/Luke/scFMs/state/experiments/cytokine_validation_2026/Onek1k/templates/per_pert"

# Optional: restrict gene space (recommended for STATE)
COMMON_GENES_PATH = None

# Input obs column names
CELLTYPE_COL = "cell_type"   # must exist if CELLTYPE_KEEP is not None
PERT_COL     = None          # None => do NOT filter baseline on perturbation/cytokine

# Baseline subset definition
CELLTYPE_KEEP = "CD14-positive monocyte"  # set None to keep all cell types
CYTOKINE_KEEP = None         # set None to keep all (only used if PERT_COL is not None)

# New test label column for STATE
TEST_PERT_COL = "pert_gene_test"

# CONTROL SETTINGS
CONTROL_LABEL = "PBS"        # set None => NO controls
CONTROL_FRACTION = 0.10      # ignored if CONTROL_LABEL is None
N_CONTROL_CELLS  = None      # ignored if CONTROL_LABEL is None
CONTROL_SEED = 42

# Optional marker columns (handy for debugging)
MARK_BASELINE_COL = "baseline_cytokine"    # set None to skip
MARK_CELLTYPE_COL = "baseline_cell_type"   # set None to skip
MARK_IS_CONTROL_COL = "is_control"         # set None to skip

# Perturbation dictionaries (where we pull pert list from)
PERT_DICT_PATHS = [
   "/iridisfs/ddnb/Luke/data/cytokine_1k1k/Parse_10M_PBMC_cytokines_CD14_Mono_aligned.h5ad",
]
DICT_PERT_COL = "cytokine"
EXCLUDE_PERTS = {"control", "non-targeting", "NT", "ntc", "PBS"}

# Optionally cap number of perts for a test run
MAX_PERTS = None   # e.g. 10 for debugging; set None for all

# Output naming
OUT_PREFIX = "CD14_Mono_1k1k__"
OUT_SUFFIX = "__template.h5ad"

# order control
REVERSE_PERT_ORDER = True   # True => Z->A, False => A->Z

# resume behaviour
SKIP_IF_EXISTS = True       # If template already exists, do not rewrite

# ============================================================
# END CONFIG
# ============================================================


def load_common_genes(path: str | None) -> list[str] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] common genes file not found: {p}")
    genes = [g.strip() for g in p.read_text().splitlines() if g.strip()]
    if not genes:
        raise ValueError(f"[ERROR] common genes file empty: {p}")
    print(f"[INFO] Loaded {len(genes):,} common genes from {p}", flush=True)
    return genes


def subset_to_genes(adata: ad.AnnData, genes: list[str] | None, label: str) -> ad.AnnData:
    if genes is None:
        return adata
    present = [g for g in genes if g in adata.var_names]
    print(f"[INFO] [{label}] genes kept: {len(present):,} / {len(genes):,}", flush=True)
    if len(present) == 0:
        raise RuntimeError(f"[ERROR] [{label}] 0 genes remain after filtering")
    return adata[:, present].copy()


def build_test_perturbation_list() -> list[str]:
    all_perts = []
    for p in PERT_DICT_PATHS:
        a = ad.read_h5ad(p)
        if DICT_PERT_COL not in a.obs.columns:
            raise KeyError(f"[ERROR] '{DICT_PERT_COL}' not found in obs of {p}")
        all_perts.append(a.obs[DICT_PERT_COL].astype(str))

    perts = pd.concat(all_perts, ignore_index=True)
    perts = perts[~perts.isna()]
    perts = perts[~perts.isin(EXCLUDE_PERTS)]
    unique = perts.drop_duplicates().tolist()

    if not unique:
        raise RuntimeError("[ERROR] No valid perturbations after filtering.")

    # Deterministic ordering, then optional reverse
    unique = sorted(unique, reverse=REVERSE_PERT_ORDER)

    if MAX_PERTS is not None:
        unique = unique[: int(MAX_PERTS)]

    direction = "Z->A" if REVERSE_PERT_ORDER else "A->Z"
    print(f"[INFO] Unique test perts: {len(unique):,} (order={direction})", flush=True)
    if len(unique) > 0:
        print(f"[INFO] First 5 perts: {unique[:5]}", flush=True)
        print(f"[INFO] Last  5 perts: {unique[-5:]}", flush=True)

    return unique


def safe_token(s: str) -> str:
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\.\+]+", "_", s)
    return s


def choose_n_controls(n_total: int) -> int:
    """Decide how many controls per template."""
    if n_total <= 0:
        return 0

    if N_CONTROL_CELLS is not None:
        n_ctrl = int(N_CONTROL_CELLS)
        if n_ctrl < 0:
            raise ValueError("[ERROR] N_CONTROL_CELLS must be >= 0")
        return min(n_ctrl, n_total)

    if CONTROL_FRACTION is None:
        raise ValueError("[ERROR] Set CONTROL_FRACTION or N_CONTROL_CELLS")

    if not (0.0 <= CONTROL_FRACTION <= 1.0):
        raise ValueError("[ERROR] CONTROL_FRACTION must be between 0 and 1")

    n_ctrl = int(round(n_total * CONTROL_FRACTION))
    n_ctrl = min(max(n_ctrl, 0), n_total)
    return n_ctrl


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_genes = load_common_genes(COMMON_GENES_PATH)

    print(f"[INFO] Loading baseline pool: {INPUT_H5AD}", flush=True)
    adata = ad.read_h5ad(INPUT_H5AD)

    # subset genes early (reduces memory per template)
    adata = subset_to_genes(adata, common_genes, label="baseline_pool")

    # -------------------------
    # Baseline subsetting (None-safe)
    # -------------------------
    if CELLTYPE_KEEP is not None:
        if CELLTYPE_COL is None:
            raise ValueError("[ERROR] CELLTYPE_COL cannot be None when CELLTYPE_KEEP is set")
        if CELLTYPE_COL not in adata.obs.columns:
            raise KeyError(f"[ERROR] '{CELLTYPE_COL}' not found in input obs")
        ct_series = adata.obs[CELLTYPE_COL].astype(str)
        m_ct = ct_series.values == str(CELLTYPE_KEEP)
    else:
        m_ct = np.ones(adata.n_obs, dtype=bool)

    # Pert/cytokine filter is OPTIONAL:
    # - If PERT_COL is None OR CYTOKINE_KEEP is None -> do NOT filter on this dimension
    if (PERT_COL is None) or (CYTOKINE_KEEP is None):
        m_p = np.ones(adata.n_obs, dtype=bool)
    else:
        if PERT_COL not in adata.obs.columns:
            raise KeyError(f"[ERROR] '{PERT_COL}' not found in input obs")
        p_series = adata.obs[PERT_COL].astype(str)
        m_p = p_series.values == str(CYTOKINE_KEEP)

    base = adata[m_ct & m_p].copy()

    if base.n_obs == 0:
        # Helpful diagnostics
        msg = ["[ERROR] baseline subset is empty after filtering."]
        msg.append(f"  CELLTYPE_COL={CELLTYPE_COL}")
        msg.append(f"  CELLTYPE_KEEP={repr(CELLTYPE_KEEP)}")
        msg.append(f"  PERT_COL={repr(PERT_COL)}")
        msg.append(f"  CYTOKINE_KEEP={repr(CYTOKINE_KEEP)}")
        if (CELLTYPE_KEEP is not None) and (CELLTYPE_COL in adata.obs.columns):
            ct = adata.obs[CELLTYPE_COL].astype(str)
            exact = int((ct.values == str(CELLTYPE_KEEP)).sum())
            msg.append(f"  Exact cell-type matches={exact}")
            top = ct.value_counts(dropna=False).head(20).to_string()
            msg.append("  Top 20 cell types:\n" + top)
        raise RuntimeError("\n".join(msg))

    print(f"[INFO] Baseline subset: {base.n_obs:,} cells × {base.n_vars:,} genes", flush=True)

    if MARK_BASELINE_COL is not None:
        base.obs[MARK_BASELINE_COL] = CYTOKINE_KEEP if CYTOKINE_KEEP is not None else "all"
    if MARK_CELLTYPE_COL is not None:
        base.obs[MARK_CELLTYPE_COL] = CELLTYPE_KEEP if CELLTYPE_KEEP is not None else "all"

    # -------------------------
    # Decide control indices ONCE so every template uses the same control cells
    # (None-safe: CONTROL_LABEL=None => no controls)
    # -------------------------
    n_total = base.n_obs
    rng = np.random.default_rng(CONTROL_SEED)

    control_mask = np.zeros(n_total, dtype=bool)

    if CONTROL_LABEL is None:
        n_ctrl = 0
        print("[INFO] CONTROL_LABEL=None => NO controls will be created.", flush=True)
    else:
        n_ctrl = choose_n_controls(n_total)
        if n_ctrl == 0:
            print(
                f"[WARN] n_ctrl=0 => templates will have NO controls. "
                f"STATE cannot use --control-pert '{CONTROL_LABEL}'.",
                flush=True,
            )
        else:
            ctrl_idx = rng.choice(n_total, size=n_ctrl, replace=False)
            control_mask[ctrl_idx] = True
            print(
                f"[INFO] Controls per template: {n_ctrl:,} / {n_total:,} "
                f"({(n_ctrl / n_total) * 100:.2f}%) with label '{CONTROL_LABEL}'",
                flush=True,
            )

    # build perturbation list (ordered deterministically)
    test_perts = build_test_perturbation_list()

    # write a manifest for your ISP scripts (in the same order you will process)
    manifest_path = out_dir / "manifest_templates.tsv"
    with open(manifest_path, "w") as mf:
        mf.write("pert\tpath\n")
        for g in test_perts:
            out_name = f"{safe_token(OUT_PREFIX)}__{safe_token(g)}{OUT_SUFFIX}"
            out_path = out_dir / out_name
            mf.write(f"{g}\t{out_path}\n")
    print(f"[INFO] Wrote manifest: {manifest_path}", flush=True)

    # write one .h5ad per perturbation
    base_obs_names = base.obs_names.to_numpy()

    wrote = 0
    skipped = 0

    for i, g in enumerate(test_perts, start=1):
        out_name = f"{safe_token(OUT_PREFIX)}__{safe_token(g)}{OUT_SUFFIX}"
        out_path = out_dir / out_name

        if SKIP_IF_EXISTS and out_path.exists():
            skipped += 1
            if skipped <= 20 or (skipped % 200 == 0):
                print(f"[SKIP] ({i}/{len(test_perts)}) Exists: {out_path.name}", flush=True)
            continue

        print(f"[INFO] ({i}/{len(test_perts)}) Writing template for: {g}", flush=True)

        tmp = base.copy()

        # Build pert labels in TEST_PERT_COL (None-safe controls)
        if CONTROL_LABEL is None:
            tmp.obs[TEST_PERT_COL] = g
        else:
            values = np.empty(tmp.n_obs, dtype=object)
            values[control_mask] = CONTROL_LABEL
            values[~control_mask] = g
            tmp.obs[TEST_PERT_COL] = values

        if MARK_IS_CONTROL_COL is not None:
            tmp.obs[MARK_IS_CONTROL_COL] = control_mask if CONTROL_LABEL is not None else False

        # make barcodes unique per perturbation file
        tmp.obs_names = np.array([f"{bc}__{safe_token(g)}" for bc in base_obs_names], dtype=object)
        if not tmp.obs_names.is_unique:
            raise RuntimeError(f"[ERROR] obs_names not unique for perturbation '{g}'")

        tmp.write_h5ad(out_path, compression="gzip")
        wrote += 1

        if wrote % 200 == 0:
            print(f"[INFO] Progress: wrote={wrote:,}, skipped={skipped:,}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"[OK] Done. Output dir: {out_dir}", flush=True)
    print(f"[OK] Templates written: {wrote:,}", flush=True)
    print(f"[OK] Templates skipped (already existed): {skipped:,}", flush=True)
    print(f"[OK] Manifest: {manifest_path}", flush=True)

    if CONTROL_LABEL is None:
        print(f"[NOTE] Run STATE inference with: --pert-col {TEST_PERT_COL}  (NO --control-pert)", flush=True)
    else:
        print(f"[NOTE] Run STATE inference with: --pert-col {TEST_PERT_COL} --control-pert {CONTROL_LABEL}", flush=True)

    print(f"[NOTE] Current order: {'Z->A' if REVERSE_PERT_ORDER else 'A->Z'}", flush=True)


if __name__ == "__main__":
    main()
