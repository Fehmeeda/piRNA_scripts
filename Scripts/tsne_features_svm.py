import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# ============================
# Load t-SNE features
# ============================
data = np.load("Human/Human_tsne_features.npz")
X = data["X"]    # shape (N, 2)
y = data["y"]

# ============================
# Train/Test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# Train SVM
# ============================
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train, y_train)

# ============================
# Predict
# ============================
y_prob = svm.predict_proba(X_test)[:, 1]
y_pred = svm.predict(X_test)

# ============================
# Evaluation
# ============================
print("\n===== SVM on t-SNE RESULTS =====\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)

# ============================
# Confusion Matrix
# ============================
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("SVM Confusion Matrix (t-SNE)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()

# ============================
# ROC Curve
# ============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("SVM ROC (t-SNE)")
plt.legend()
plt.show()
