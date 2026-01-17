import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

species_files = {
    "Human": "Human/Human_kmer_weighted_onehot_with_motif_weight_1.npz",
    "Mouse": "Mouse/Mouse_kmer_weighted_onehot_with_motif_weight_1.npz",
    "Drosophila": "Drosophila/Drosophila_kmer_weighted_onehot_with_motif_weight_1.npz"
}

for species, file_path in species_files.items():
    print(f"\n=== Processing {species} ===")

    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    # Flatten
    X_flat = np.array([x.flatten() for x in X])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # PCA (FEATURES for ML)
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("PCA shape:", X_pca.shape)

    # Save PCA features
    np.savez(
        f"{species}/{species}_pca10_features_kmer_weighted_onehot_with_motif_weight_1.npz",
        X=X_pca,
        y=y
    )
