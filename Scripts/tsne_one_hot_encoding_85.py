import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# =============================
# List the files you want to load
# =============================
species_files = {
    "Human": "Human/Human_kmer_onehot_85.npz",
    "Mouse": "Mouse/Mouse_kmer_onehot_85.npz",
    "Drosophila": "Drosophila/Drosophila_kmer_onehot_85.npz"
}

# =============================
# Process each species separately
# =============================
for species, file_path in species_files.items():
    print(f"\n=== Processing {species} ===")
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    X = data['X']  # shape: (num_sequences, 125, kmer)
    y = data['y']
    
    print(f"Loaded {species}: X={X.shape}, y={y.shape}")

    # Flatten one-hot matrices
    X_flat = np.array([x.flatten() for x in X])
    print(f"Flattened shape: {X_flat.shape}")

    # Run t-SNE
    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_flat)

    # Plot positive vs negative
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='blue', label='Positive', alpha=0.6)
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='red', label='Negative', alpha=0.6)
    plt.title(f"t-SNE of One-hot Encoded Sequences (Overlapping) — {species}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{species}/tSNE_onehot_overlapping_{species}.png")
    plt.show()

