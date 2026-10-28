# ============================================================
# Cytokine classification probe on STATE embeddings
# Logistic Regression + Gradient Boosting
# Loads AnnData, saves reports + confusion matrices
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

# -----------------------
# CONFIG
# -----------------------
h5ad_path = Path(
    "/iridisfs/ddnb/Luke/scFMs/state/experiments/replogle_cytokine_experiment/run1_2025_12_05/embeddings/Parse_10M_CD14Mono_state_aligned_embedding.h5ad"   # <-- CHANGE IF NEEDED
)

emb_key = "X_state"        # embedding key in adata.obsm
label_col = "pert_gene"     # cytokine / pert column (PBS stays "PBS")

output_dir = Path(
    "/iridisfs/ddnb/Luke/scFMs/state/experiments/"
    "replogle_cytokine_experiment/run1_2025_12_05/embeddings"
)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading AnnData from: {h5ad_path}")
print(f"Saving outputs to:   {output_dir}")

# -----------------------
# LOAD ANNDATA
# -----------------------
adata = sc.read_h5ad(h5ad_path)

# Sanity checks
assert emb_key in adata.obsm, f"{emb_key} not found in adata.obsm"
assert label_col in adata.obs, f"{label_col} not found in adata.obs"

# -----------------------
# EXTRACT DATA
# -----------------------
X = adata.obsm[emb_key]
y = adata.obs[label_col].astype(str).values

le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"Classes ({len(le.classes_)}): {le.classes_}")

# -----------------------
# TRAIN / TEST SPLIT
# -----------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# -----------------------
# HELPER FUNCTION
# -----------------------
def save_cm_plot(cm, labels, outpath, title, normalize=False):
    plt.figure(figsize=(6, 5))
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        annot=True,
        fmt=fmt
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ============================================================
# 1) LOGISTIC REGRESSION (LINEAR PROBE)
# ============================================================
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

logreg.fit(X_tr, y_tr)
pred_lr = logreg.predict(X_te)

bal_acc_lr = balanced_accuracy_score(y_te, pred_lr)
report_lr = classification_report(
    y_te, pred_lr, target_names=le.classes_
)

with open(output_dir / "classification_report_logreg.txt", "w") as f:
    f.write(f"Balanced accuracy: {bal_acc_lr:.4f}\n\n")
    f.write(report_lr)

cm_lr = confusion_matrix(y_te, pred_lr)
pd.DataFrame(
    cm_lr,
    index=le.classes_,
    columns=le.classes_
).to_csv(output_dir / "confusion_matrix_logreg.csv")

save_cm_plot(
    cm_lr,
    le.classes_,
    output_dir / "confusion_matrix_logreg.png",
    title="Logistic Regression – Confusion Matrix"
)

save_cm_plot(
    cm_lr,
    le.classes_,
    output_dir / "confusion_matrix_logreg_normalized.png",
    title="Logistic Regression – Normalized Confusion Matrix",
    normalize=True
)

# ============================================================
# 2) GRADIENT BOOSTING
# ============================================================
gb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.1,
    random_state=0
)

gb.fit(X_tr, y_tr)
pred_gb = gb.predict(X_te)

bal_acc_gb = balanced_accuracy_score(y_te, pred_gb)
report_gb = classification_report(
    y_te, pred_gb, target_names=le.classes_
)

with open(output_dir / "classification_report_gb.txt", "w") as f:
    f.write(f"Balanced accuracy: {bal_acc_gb:.4f}\n\n")
    f.write(report_gb)

cm_gb = confusion_matrix(y_te, pred_gb)
pd.DataFrame(
    cm_gb,
    index=le.classes_,
    columns=le.classes_
).to_csv(output_dir / "confusion_matrix_gb.csv")

save_cm_plot(
    cm_gb,
    le.classes_,
    output_dir / "confusion_matrix_gb.png",
    title="Gradient Boosting – Confusion Matrix"
)

save_cm_plot(
    cm_gb,
    le.classes_,
    output_dir / "confusion_matrix_gb_normalized.png",
    title="Gradient Boosting – Normalized Confusion Matrix",
    normalize=True
)

print("Done. Reports and confusion matrices saved.")
