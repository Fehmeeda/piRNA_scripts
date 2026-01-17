import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ============================
# Load t-SNE features
# ============================
data = np.load("Human/Human_tsne_features.npz")
X = data["X"]   # shape (N, 2)
y = data["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================
# Train / Test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# Build NN (small network for 2D input)
# ============================
model = Sequential([
    Dense(64, activation="relu", input_shape=(2,)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# Train
# ============================
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ============================
# Predict
# ============================
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# ============================
# Evaluation
# ============================
print("\n===== NN on t-SNE RESULTS =====\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", roc_auc)

# ============================
# Confusion Matrix
# ============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap="Greens")
plt.title("NN Confusion Matrix (t-SNE)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# ============================
# ROC Curve
# ============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("NN ROC Curve (t-SNE)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================
# Training & Validation Curves
# ============================
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("NN Loss Curve (t-SNE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("NN Accuracy Curve (t-SNE)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
