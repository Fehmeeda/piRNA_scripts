import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ==================================
# Files to load (dna2vec embeddings)
# ==================================
species_files = {
    "Human": "Human/Human_dna2vec_embeddings.npz",
    "Mouse": "Mouse/Mouse_dna2vec_embeddings.npz",
    "Drosophila": "Drosophila/Drosophila_dna2vec_embeddings.npz"
}

# ==================================
# Run t-SNE for each species
# ==================================
for species, file_path in species_files.items():
    print(f"\n=== Processing {species} ===")

    # Load data
    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    print(f"{X.shape}")
    #print(X[0][23])      # shape: (num_seq, max_kmers, 100)
    y = data["y"]

    print(f"Loaded {species}: X={X.shape}, y={y.shape}")

    # Flatten dna2vec matrices
    X_flat = np.array([x.flatten() for x in X])
    print(f"Flattened shape: {X_flat.shape}")
    #print(X_flat[0:101])

    # Run t-SNE (recommended settings)
    tsne = TSNE()

    X_tsne = tsne.fit_transform(X_flat)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1],
                c='blue', label='Positive', alpha=0.6)
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1],
                c='red', label='Negative', alpha=0.6)

    plt.title(f"t-SNE of dna2vec Embeddings using overlapping Kmer — {species}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{species}/tSNE_dna2vec_overlapping{species}.png", dpi=300)
    plt.show()
