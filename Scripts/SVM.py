'''import numpy as np

X = np.load("Human_X_kmer_prob.npy")
y = np.load("Human_y.npy")

print(X.shape, y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("SVM AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

nn.fit(X_train, y_train)

y_pred_nn = nn.predict(X_test)
y_prob_nn = nn.predict_proba(X_test)[:, 1]

print("NN Accuracy:", accuracy_score(y_test, y_pred_nn))
print("NN AUC:", roc_auc_score(y_test, y_prob_nn))
print(classification_report(y_test, y_pred_nn))

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(svm, X, y, cv=cv, scoring="roc_auc")

print("SVM CV AUC:", scores.mean(), "±", scores.std())
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ============================
# Species data
# ============================
species_files = {
    "Human": ("Human_X_kmer_prob.npy", "Human_y.npy"),
    "Mouse": ("Mouse_X_kmer_prob.npy", "Mouse_y.npy"),
    "Drosophila": ("Drosophila_X_kmer_prob.npy", "Drosophila_y.npy")
}

# ============================
# Output directory
# ============================
BASE_OUTDIR = "SVM_Results_RBF_kernel_c=10"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# ============================
# Loop over species
# ============================
for species, (X_file, y_file) in species_files.items():
    print(f"\n================ {species} =================")

    outdir = os.path.join(BASE_OUTDIR, species)
    os.makedirs(outdir, exist_ok=True)

    # ----------------------------
    # Load data
    # ----------------------------
    X = np.load(X_file)
    #print(X.shape)
    y = np.load(y_file)

    # ----------------------------
    # Train / test split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ----------------------------
    # SVM Pipeline
    # ----------------------------
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    # ----------------------------
    # Train
    # ----------------------------
    svm_model.fit(X_train, y_train)

    # ----------------------------
    # Predict
    # ----------------------------
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]

    # ----------------------------
    # Metrics
    # ----------------------------
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(report)

    # ----------------------------
    # Save text results
    # ----------------------------
    with open(os.path.join(outdir, "svm_report_linear.txt"), "w") as f:
        f.write(f"{species} SVM Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write(report)

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{species} SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()

    # ----------------------------
    # ROC Curve
    # ----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{species} SVM ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close()

    print(f"Saved all SVM results for {species} → {outdir}")
