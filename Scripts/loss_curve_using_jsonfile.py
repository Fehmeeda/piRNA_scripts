import json
import matplotlib.pyplot as plt
import os
import glob

SPECIES = ["Human", "Mouse", "Drosophila"]
FOLDS = range(5)

def plot_loss_curves(json_path, title=None, save_path=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


# ============================================================
# MAIN LOOP: all result folders, species, folds
# ============================================================

results_dirs = glob.glob("results_cnn_*")

for results_dir in results_dirs:
    print(f"Processing {results_dir}")

    for species in SPECIES:
        for fold in FOLDS:
            pattern = f"{results_dir}/{species}_fold{fold}_*.json"
            json_files = glob.glob(pattern)

            if not json_files:
                continue  # skip missing folds/species

            json_path = json_files[0]

            save_path = f"{results_dir}/{species}_fold{fold}_loss.png"
            title = f"{species} Fold {fold} Training vs Validation Loss"

            plot_loss_curves(
                json_path=json_path,
                title=title,
                save_path=save_path
            )
