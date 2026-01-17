import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# =============================
# Species files
# =============================
species_files = {
    "Human": "Human/Human_dna2vec_embeddings.npz",
    "Mouse": "Mouse/Mouse_dna2vec_embeddings.npz",
    "Drosophila": "Drosophila/Drosophila_dna2vec_embeddings.npz"
}

# =============================
# Output directory
# =============================
BASE_OUTDIR = "NN_dna2vec_32_16_dropout_0.5_learning_0.005_epoch_100"
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
    X = data["X"]
    X = data["X"]   
    print(X.shape)
    X = np.array([x.flatten() for x in X])
    print(f"Flattened shape: {X.shape}")
    y = data["y"]

    # -----------------------------
    # Train / test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # Standardize
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Neural Network
    # -----------------------------
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(16, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # -----------------------------
    # Predict
    # -----------------------------
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(report)
    print("ROC-AUC:", roc_auc)

    # -----------------------------
    # Save text report
    # -----------------------------
    with open(os.path.join(outdir, "nn_dna2vec.txt"), "w") as f:
        f.write(f"{species} — Neural Network on dna2vec\n")
        f.write("=" * 50 + "\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write(report)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Greens")
    plt.title(f"{species} NN Confusion Matrix")
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
    plt.title(f"{species} NN ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close()

    # -----------------------------
    # Loss Curve
    # -----------------------------
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{species} NN Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    # -----------------------------
    # Accuracy Curve
    # -----------------------------
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{species} NN Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()

    print(f"Saved all NN results → {outdir}")
