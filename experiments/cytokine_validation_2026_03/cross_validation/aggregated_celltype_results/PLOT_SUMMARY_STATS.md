# CV Validation Plot Summary Statistics

Use these exact numbers when writing about Figures A–E. Data: 9-fold leave-one-cytokine-out CV, mean per (cell type × cytokine) across folds.

---

## Data overview
- **Folds**: 9
- **Cell types**: 17
- **Cytokines**: 90
- **Conditions** (cell type × cytokine): 1480

---

## A) PR-AUC heatmap

### Overall
- Mean: 0.257
- Median: 0.215
- SD: 0.211
- Range: [0.000–1.000]
- IQR (Q25–Q75): 0.077–0.407

### Best / worst cell type (by median PR-AUC)
- Best: **CD14 Mono** (median = 0.563)
- Worst: **ILC** (median = 0.001)

### Best / worst cytokine (by median PR-AUC)
- Best: **IFN-beta** (median = 0.629)
- Worst: **VEGF** (median = 0.047)

---

## B) Pearson delta heatmap

### Overall
- Mean: 0.357
- Median: 0.384
- SD: 0.240
- Range: [-0.384–0.914]
- IQR (Q25–Q75): 0.185–0.525

### Best / worst cell type (by median pearson_delta)
- Best: **CD4 Naive** (median = 0.522)
- Worst: **Plasmablast** (median = 0.027)

### Best / worst cytokine (by median pearson_delta)
- Best: **IFN-beta** (median = 0.818)
- Worst: **M-CSF** (median = -0.013)

---

## C) PR-AUC box plot (by cell type)

Same statistics as A, displayed as box plot per cell type. Cell types ordered by median PR-AUC (descending).

### Per-cell-type medians (for quick reference)

- CD14 Mono: median = 0.563, IQR = 0.186
- CD4 Memory: median = 0.495, IQR = 0.250
- CD4 Naive: median = 0.445, IQR = 0.204
- NK: median = 0.305, IQR = 0.215
- B Naive: median = 0.304, IQR = 0.201
- CD16 Mono: median = 0.287, IQR = 0.189
- CD8 Memory: median = 0.285, IQR = 0.257
- CD8 Naive: median = 0.187, IQR = 0.136
- MAIT: median = 0.174, IQR = 0.209
- cDC: median = 0.151, IQR = 0.180
- Treg: median = 0.118, IQR = 0.174
- NKT: median = 0.097, IQR = 0.129
- NK CD56bright: median = 0.087, IQR = 0.161
- pDC: median = 0.035, IQR = 0.077
- HSPC: median = 0.010, IQR = 0.120
- Plasmablast: median = 0.001, IQR = 0.002
- ILC: median = 0.001, IQR = 0.009

---

## D) Pearson delta box plot (by cell type)

Same statistics as B, displayed as box plot per cell type. Cell types ordered by median pearson_delta (descending).

### Per-cell-type medians (for quick reference)

- CD4 Naive: median = 0.522, IQR = 0.348
- CD4 Memory: median = 0.493, IQR = 0.220
- B Naive: median = 0.492, IQR = 0.207
- CD8 Memory: median = 0.485, IQR = 0.272
- NK: median = 0.480, IQR = 0.284
- CD14 Mono: median = 0.452, IQR = 0.288
- MAIT: median = 0.431, IQR = 0.215
- CD16 Mono: median = 0.421, IQR = 0.215
- Treg: median = 0.387, IQR = 0.317
- NK CD56bright: median = 0.353, IQR = 0.224
- cDC: median = 0.347, IQR = 0.303
- CD8 Naive: median = 0.346, IQR = 0.292
- NKT: median = 0.317, IQR = 0.279
- pDC: median = 0.166, IQR = 0.244
- ILC: median = 0.115, IQR = 0.325
- HSPC: median = 0.110, IQR = 0.184
- Plasmablast: median = 0.027, IQR = 0.147

---

## E) Pearson delta vs PR-AUC scatter

### Correlation between metrics
- Pearson r = 0.553 (p < 0.001)
- Spearman ρ = 0.603 (p < 0.001)

### Top 10 conditions (by combined score = (pearson_delta + pr_auc) / 2)

1. CD14 Mono – IFN-beta (pr_auc=0.826, r=0.914)
2. CD14 Mono – IL-4 (pr_auc=0.827, r=0.885)
3. CD14 Mono – IFN-gamma (pr_auc=0.781, r=0.885)
4. cDC – IFN-beta (pr_auc=0.780, r=0.864)
5. ILC – CD40L (pr_auc=1.000, r=0.642)
6. cDC – IFN-gamma (pr_auc=0.775, r=0.827)
7. CD14 Mono – IL-13 (pr_auc=0.855, r=0.740)
8. CD14 Mono – IL-2 (pr_auc=0.780, r=0.795)
9. CD14 Mono – IFN-omega (pr_auc=0.859, r=0.701)
10. B Naive – IFN-beta (pr_auc=0.652, r=0.902)

---

## Figure → file mapping
| Panel | Plot | File |
|-------|------|------|
| A | PR-AUC heatmap | `heatmap_pr_auc.png` |
| B | Pearson delta heatmap | `heatmap_pearson_delta.png` |
| C | PR-AUC box plot | `plots/celltype_box_pr_auc.png` |
| D | Pearson delta box plot | `plots/celltype_box_pearson_delta.png` |
| E | pearson_delta vs pr_auc scatter | `plots/scatter_pearson_vs_pr_auc.png` |
