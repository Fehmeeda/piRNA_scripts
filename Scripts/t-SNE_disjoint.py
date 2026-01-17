import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------------
# Species and their files
# -----------------------------
species_files = {
    "Human": ("Human/Human_X_pos_disjoint.npy", "Human/Human_X_neg_disjoint.npy"),
    "Mouse": ("Mouse/Mouse_X_pos_disjoint.npy", "Mouse/Mouse_X_neg_disjoint.npy"),
    "Drosophila": ("Drosophila/Drosophila_X_pos_disjoint.npy", "Drosophila/Drosophila_X_neg_disjoint.npy")
}

# -----------------------------
# Process each species
# -----------------------------
for species, (pos_file, neg_file) in species_files.items():
    print(f"\nProcessing {species}...")

    # Load positive and negative sequences
    X_pos = np.load(pos_file, allow_pickle=True)
    X_neg = np.load(neg_file, allow_pickle=True)
    print(f"{species}: Positive={len(X_pos)}, Negative={len(X_neg)}")

    # Flatten sequences
    X_pos_flat = np.array([x.flatten() for x in X_pos])
    X_neg_flat = np.array([x.flatten() for x in X_neg])

    # Combine
    X_all_flat = np.vstack([X_pos_flat, X_neg_flat])
    y_all = np.array([1]*len(X_pos_flat) + [0]*len(X_neg_flat))

    # t-SNE
    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_all_flat)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[y_all==1,0], X_tsne[y_all==1,1], c='blue', label='Positive', alpha=0.6)
    plt.scatter(X_tsne[y_all==0,0], X_tsne[y_all==0,1], c='red', label='Negative', alpha=0.6)
    plt.title(f"t-SNE of {species} Disjoint 3-mer One-hot Sequences")
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(f"{species}/{species}_tsne_disjoint.png", dpi=300)
    plt.show()
    print(f"Saved t-SNE plot for {species}")
