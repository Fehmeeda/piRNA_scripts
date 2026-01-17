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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ============================
# Species data paths
# ============================
species_files = {
    "Human": ("Human_X_kmer_prob.npy", "Human_y.npy"),
    "Mouse": ("Mouse_X_kmer_prob.npy", "Mouse_y.npy"),
    "Drosophila": ("Drosophila_X_kmer_prob.npy", "Drosophila_y.npy")
}

# ============================
# Output directory
# ============================
BASE_OUTDIR = "NN_Results_kmer_prob_without_early_stop_epoch_100_complexity_64_32_dropout_0.3_learning_rate_0.001"
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
    # Standardize
    # ----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ----------------------------
    # Build NN
    # ----------------------------
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    '''
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    '''
    # ----------------------------
    # Train
    # ----------------------------
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        #callbacks=[early_stop],
        verbose=1
    )

    # ----------------------------
    # Predict
    # ----------------------------
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # ----------------------------
    # Evaluation
    # ----------------------------
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(report)
    print("ROC-AUC:", roc_auc)

    # Save report
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(f"{species} Neural Network Results\n\n")
        f.write(report)
        f.write(f"\nROC-AUC: {roc_auc:.4f}\n")

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
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

    # ----------------------------
    # ROC Curve
    # ----------------------------
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

    # ----------------------------
    # Loss curve
    # ----------------------------
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{species} NN Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    # ----------------------------
    # Accuracy curve
    # ----------------------------
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{species} NN Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()

    print(f"Saved all results for {species} → {outdir}")
