import numpy as np
import umap
import matplotlib.pyplot as plt

# =============================
# List the files you want to load
# =============================
species_files = {
    "Human": "Human/Human_kmer_weighted_onehot.npz",
    "Mouse": "Mouse/Mouse_kmer_weighted_onehot.npz",
    "Drosophila": "Drosophila/Drosophila_kmer_weighted_onehot.npz"
}

# =============================
# Process each species separately
# =============================
for species, file_path in species_files.items():
    print(f"\n=== Processing {species} ===")

    # Load data
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]  # (num_sequences, 125, kmer)
    y = data["y"]

    print(f"Loaded {species}: X={X.shape}, y={y.shape}")

    # Flatten one-hot matrices
    X_flat = np.array([x.flatten() for x in X])
    print(f"Flattened shape: {X_flat.shape}")

    # =============================
    # UMAP
    # =============================
    reducer = umap.UMAP(n_neighbors=50)

    X_umap = reducer.fit_transform(X_flat)
    print(f"UMAP shape: {X_umap.shape}")

    # =============================
    # Plot
    # =============================
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_umap[y == 0, 0],
        X_umap[y == 0, 1],
        c="red",
        label="Negative",
        alpha=0.6,
        s=20
    )

    plt.scatter(
        X_umap[y == 1, 0],
        X_umap[y == 1, 1],
        c="blue",
        label="Positive",
        alpha=0.6,
        s=20
    )


    plt.title(f"UMAP of Weighted One-hot Encoded Sequences — {species}")
    plt.legend()
    plt.tight_layout()

    out_png = f"{species}/UMAP_weighted_onehot_{species}.png"
    plt.savefig(out_png, dpi=300)
    plt.show()

    # =============================
    # Save UMAP features
    # =============================
    np.savez(
        f"{species}/{species}_umap_features_kmer_weighted_onehot.npz",
        X=X_umap,
        y=y
    )

    print(f"Saved UMAP plot: {out_png}")
