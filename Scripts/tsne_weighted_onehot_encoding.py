import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# =============================
# List the files you want to load
# =============================
species_files = {
    #"Human": "Human/Human_kmer_weighted_onehot_with_motif_weight_1.npz",
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
    X = data['X']  # shape: (num_sequences, 125, kmer)
    print(X.shape)
    y = data['y']
    
    print(f"Loaded {species}: X={X.shape}, y={y.shape}")

    # Flatten one-hot matrices
    X_flat = np.array([x.flatten() for x in X])
    print(f"Flattened shape: {X_flat.shape}")

    # Run t-SNE
    tsne = TSNE(perplexity=30,random_state=42)
    X_tsne = tsne.fit_transform(X_flat)
    
    # Plot positive vs negative
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='blue', label='Positive', alpha=0.6)
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='red', label='Negative', alpha=0.6)
    plt.title(f"t-SNE of weighted One-hot Encoded Sequences (Overlapping) — {species}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{species}/tSNE_weighted_onehot_overlapping_{species}.png")
    plt.show()
    
    print(X_tsne.shape)
    print(X_tsne[1])
    np.savez(
    f"{species}/{species}tsne_features_kmer_weighted_onehot.npz",
    X=X_tsne,
    y=y
)
'''
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# =============================
# Species files
# =============================
species_files = {
    "Human": "Human/Human_kmer_weighted_onehot.npz",
    "Mouse": "Mouse/Mouse_kmer_weighted_onehot.npz",
    "Drosophila": "Drosophila/Drosophila_kmer_weighted_onehot.npz"
}

# =============================
# Process each species
# =============================
for species, file_path in species_files.items():
    print(f"\n=== Processing {species} ===")

    data = np.load(file_path, allow_pickle=True)
    X = data["X"]   # (N, 125, kmer)
    y = data["y"]

    print(f"Loaded {species}: X={X.shape}, y={y.shape}")

    # Flatten
    X_flat = X.reshape(X.shape[0], -1)
    print(f"Flattened shape: {X_flat.shape}")

    # t-SNE
    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_flat)

    # =============================
    # Separate plots
    # =============================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Positive samples
    axes[0].scatter(
        X_tsne[y == 1, 0],
        X_tsne[y == 1, 1],
        c="blue",
        alpha=0.6
    )
    axes[0].set_title(f"{species} — Positive samples")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # Negative samples
    axes[1].scatter(
        X_tsne[y == 0, 0],
        X_tsne[y == 0, 1],
        c="red",
        alpha=0.6
    )
    axes[1].set_title(f"{species} — Negative samples")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    plt.suptitle(f"t-SNE Visualization of Weighted One-hot Features — {species}")
    plt.tight_layout()
    plt.savefig(f"{species}/tSNE_separate_pos_neg_{species}.png")
    plt.show()

    # =============================
    # Save embeddings
    # =============================
    np.savez(
        f"{species}/{species}_tsne_features.npz",
        X=X_tsne,
        y=y
    )

    print(f"Saved: {species}_tsne_features.npz")
'''
