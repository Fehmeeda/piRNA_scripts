import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# =============================
# Species files
# =============================
species_files = {
    "Human": "Human/Human_tsne_features_kmer_weighted_onehot_with_motif_weight_2.npz",
    "Mouse": "Mouse/Mouse_tsne_features_kmer_weighted_onehot_with_motif_weight_2.npz",
    "Drosophila": "Drosophila/Drosophila_tsne_features_kmer_weighted_onehot_with_motif_weight_2.npz"
}

# =============================
# Output base directory
# =============================
BASE_OUTDIR = "SVM_tSNE_Results_weight_2_linear"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# =============================
# Loop over species
# =============================
for species, file_path in species_files.items():
    print(f"\n================ {species} =================")

    outdir = os.path.join(BASE_OUTDIR, species)
    os.makedirs(outdir, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]   # (N, 2) t-SNE features
    y = data["y"]

    # -----------------------------
    # Train / Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # SVM
    # -----------------------------
    svm = SVC(
        kernel="linear",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42
    )

    svm.fit(X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("ROC-AUC:", roc_auc)
    print(report)

    # -----------------------------
    # Save text results
    # -----------------------------
    with open(os.path.join(outdir, "svm_tsne_report.txt"), "w") as f:
        f.write(f"{species} — SVM on t-SNE features\n")
        f.write("=" * 45 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write(report)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{species} — SVM (t-SNE)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{species} — ROC (t-SNE + SVM)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close()

    print(f"Saved all results → {outdir}")
