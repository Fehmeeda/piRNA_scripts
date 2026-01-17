import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ==============================
# Load Graph Data
# ==============================
def load_graph_data(npz_file):
    data = np.load(npz_file)
    adj = data["adj"]      # shape: (num_samples, K, K)
    y = data["y"]          # labels
    return adj, y

# ==============================
# Flatten adjacency matrices
# ==============================
def flatten_graphs(adj):
    # reshape each K×K matrix into 1D vector
    return adj.reshape(adj.shape[0], -1)

# ==============================
# Run TSNE
# ==============================
def run_tsne(features, labels, species_name):
    print(f"Running t-SNE for {species_name} ...")
    tsne = TSNE()

    X_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.7,
        s=20
    )
    plt.colorbar(label="Label (0=neg, 1=pos)")
    plt.title(f"t-SNE of Weighted Graphs: {species_name}")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    plt.savefig(f"{species_name}/{species_name}_tsne_graph.png", dpi=300)
    plt.show()

    print(f"Saved: {species_name}_tsne_graph.png")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    species_files = {
        "Human": "Human/Human_graph_weighted.npz",
        "Mouse": "Mouse/Mouse_graph_weighted.npz",
        "Drosophila": "Drosophila/Drosophila_graph_weighted.npz"
    }

    for sp, file in species_files.items():
        adj, y = load_graph_data(file)
        X = flatten_graphs(adj)
        run_tsne(X, y, sp)
