## Phase 1 Directionality Refocus — Short Report

### What was done
- Recomputed **fresh DE gene tables** (no reuse of prior DE outputs) for:
  - **real external**: cytokine vs control
  - **predicted external**: cytokine vs control
- Datasets:
  - `GSE96583` (IFN-beta)
  - `GSE178429` (IFN-gamma)
- Output root:
  - `experiments/cytokine_validation_2026_03/biological_coherence/de_directionality_2026-04-16`
- Key artifacts:
  - `de_tables/de_real_all.csv`
  - `de_tables/de_pred_all.csv`
  - per-condition `de_real_*`, `de_pred_*`, `de_merged_*`
  - `directionality_metrics/directionality_metrics_all_conditions.csv`
  - `directionality_metrics/directionality_sign_match_primary_vs_de.png`
  - `directionality_metrics/atlas_vs_real_external_dataset_directionality_all_conditions.csv`
  - `directionality_metrics/atlas_vs_real_external_dataset_sign_match.png`
  - `directionality_metrics/atlas_vs_predicted_external_dataset_directionality_all_conditions.csv`
  - `directionality_metrics/atlas_vs_predicted_external_dataset_sign_match.png`
  - `manifest.yaml`

### Directionality framework used
- **Primary**: all-genes sign match (up/down direction across all shared genes).
- **Secondary**: DE-subset sign match (real-FDR<0.05 plus union/intersection side metrics).
- Non-zero-only sign match reported as auxiliary.

### Main results (predicted vs real external)
- Conditions evaluated: **19**
- Mean all-genes sign match: **0.3870**
- Mean nonzero-genes sign match: **0.4857**
- Mean DE-real sign match: **0.5496**
- All-genes sign-match range across conditions: **0.2889 to 0.5906**
- Skipped conditions: **4** (all in GSE178429 due to insufficient cells / DE failure):
  - B Intermediate/Memory, B Naive, HSPC, Plasmablast

### Added atlas vs real external dataset directionality (requested)
- Computed a similar directionality table comparing **atlas DE** vs **real external dataset DE**:
  - `directionality_metrics/atlas_vs_real_external_dataset_directionality_all_conditions.csv`
  - `directionality_metrics/atlas_vs_real_external_dataset_sign_match.png`
- Conditions evaluated: **19**
- Mean all-genes sign match (atlas vs real external dataset): **0.3102**
- Mean DE-real sign match (atlas vs real external dataset): **0.6865**
- All-genes sign-match range: **0.1295 to 0.5019**

### Atlas vs predicted external dataset directionality
- Same sign-concordance framework as atlas vs real external dataset, but **atlas DE vs predicted-external DE** (`de_pred_all.csv`), per dataset × cell type × cytokine (script: `step10c_atlas_vs_predicted_external_directionality.py`).
- Artifacts:
  - `directionality_metrics/atlas_vs_predicted_external_dataset_directionality_all_conditions.csv`
  - `directionality_metrics/atlas_vs_predicted_external_dataset_sign_match.png`
- Conditions evaluated: **19**
- Mean all-genes sign match (atlas vs predicted external dataset): **0.4728**
- Mean nonzero-genes sign match: **0.5647**
- Mean DE-predicted sign match (predicted external FDR<0.05): **0.7858**
- All-genes sign-match range: **0.2733 to 0.6800**

### Notes
- Scanpy emitted repeated `invalid value encountered in log2` warnings during DE; run completed with exit code 0 and outputs were produced.
- For atlas vs external dataset comparisons (real or predicted), effect direction used `logFC` with fallback to DE score where `logFC` was missing.

### Regenerate atlas directionality plots (same date folder)
From `biological_coherence/`:

`/iridisfs/ddnb/Luke/scFMs/state/.venv/bin/python step10b_atlas_directionality.py --date-tag 2026-04-16`

`/iridisfs/ddnb/Luke/scFMs/state/.venv/bin/python step10c_atlas_vs_predicted_external_directionality.py --date-tag 2026-04-16`
