import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# ============================
# Load data
# ============================
X = np.load("Human_X_kmer_prob.npy")
y = np.load("Human_y.npy")

# ============================
# 5-fold CV setup
# ============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1, gamma="scale", probability=True))
])

acc_scores = []
auc_scores = []
roc_curves = []

# ============================
# Cross-validation
# ============================
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")

    acc_scores.append(acc)
    auc_scores.append(auc)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves.append((fpr, tpr))

# ============================
# ROC Curves (Fold-wise)
# ============================
plt.figure(figsize=(7,6))
for i, (fpr, tpr) in enumerate(roc_curves):
    plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC={auc_scores[i]:.3f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curves (5-Fold CV)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================
# Bar plots
# ============================
folds = np.arange(1,6)

plt.figure()
plt.bar(folds, acc_scores)
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy per Fold")
plt.show()

plt.figure()
plt.bar(folds, auc_scores)
plt.xlabel("Fold")
plt.ylabel("ROC-AUC")
plt.title("SVM ROC-AUC per Fold")
plt.show()

# ============================
# Final report
# ============================
print("\n===== SVM 5-FOLD RESULTS =====")
print(f"Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"ROC-AUC : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
