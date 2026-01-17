import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ============================
# Reproducibility
# ============================
np.random.seed(42)
tf.random.set_seed(42)

# ============================
# Load data
# ============================
X = np.load("Human_X_kmer_prob.npy")
y = np.load("Human_y.npy")

# ============================
# Model builder
# ============================
def build_model(input_dim):
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================
# 5-fold CV
# ============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
auc_scores = []
roc_curves = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X.shape[1])

    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    y_prob = model.predict(X_test).ravel()
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
# ROC Curves
# ============================
plt.figure(figsize=(7,6))
for i, (fpr, tpr) in enumerate(roc_curves):
    plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC={auc_scores[i]:.3f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("NN ROC Curves (5-Fold CV)")
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
plt.title("NN Accuracy per Fold")
plt.show()

plt.figure()
plt.bar(folds, auc_scores)
plt.xlabel("Fold")
plt.ylabel("ROC-AUC")
plt.title("NN ROC-AUC per Fold")
plt.show()

# ============================
# Final report
# ============================
print("\n===== NN 5-FOLD RESULTS =====")
print(f"Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"ROC-AUC : {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
