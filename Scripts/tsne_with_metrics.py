import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import csv

# --- Load dna2vec features ---
data = np.load('pirna_k-mer_onehot_encoding.npz', allow_pickle=True)
X = data['X']
y = data['y']

# --- Flatten if necessary ---
if len(X.shape) > 2:
    X_flat = X.reshape(X.shape[0], -1)
else:
    X_flat = X

# --- Random seeds ---
#random_seeds = [7, 13, 21, 42, 77, 88, 99, 123, 256, 321]
random_seeds = [77, 88, 99, 101, 281, 321, 731, 941, 1151, 2161]
# --- Create folder to save results ---
save_dir = "tsne_results_metrices"
os.makedirs(save_dir, exist_ok=True)

# --- Function to calculate cluster metrics ---
def cluster_metrics(X_emb, y_labels):
    labels = np.unique(y_labels)
    # WCSS
    wcss = 0
    centroids = {}
    for label in labels:
        cluster_points = X_emb[y_labels == label]
        centroid = cluster_points.mean(axis=0)
        centroids[label] = centroid
        wcss += ((cluster_points - centroid)**2).sum()
    # TSS
    global_mean = X_emb.mean(axis=0)
    tss = ((X_emb - global_mean)**2).sum()
    # BCSS
    bcss = tss - wcss
    # Dunn Index
    max_intra = 0
    for label in labels:
        cluster_points = X_emb[y_labels == label]
        if len(cluster_points) > 1:
            dists = pdist(cluster_points)
            max_intra = max(max_intra, dists.max())
    cluster0 = X_emb[y_labels == labels[0]]
    cluster1 = X_emb[y_labels == labels[1]]
    inter_dists = np.linalg.norm(cluster0[:, None, :] - cluster1[None, :, :], axis=2)
    min_inter = inter_dists.min()
    dunn_index = min_inter / max_intra if max_intra != 0 else np.nan
    return wcss, bcss, tss, dunn_index

# --- Prepare CSV to save metrics ---
csv_path = os.path.join(save_dir, "tsne_cluster_metrics.csv")
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Seed", "WCSS", "BCSS", "TSS", "Dunn Index"])

# --- Run t-SNE for each seed ---
for seed in random_seeds:
    print(f"Running t-SNE with random seed {seed} ...")

    tsne = TSNE(
        n_components=2,
        max_iter=3000,
        init='random',
        perplexity=50,
        random_state=seed,
        learning_rate=200
    )
    X_tsne = tsne.fit_transform(X_flat)

    # --- Compute cluster metrics ---
    wcss, bcss, tss, dunn_index = cluster_metrics(X_tsne, y)
    print("=== Cluster Metrics ===")
    print("Within-Cluster Sum of Squares (WCSS):", wcss)
    print("Between-Cluster Sum of Squares (BCSS):", bcss)
    print("Total Sum of Squares (TSS):", tss)
    print("Dunn Index:", dunn_index)
    print("=======================")

    # --- Save metrics to CSV ---
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([seed, wcss, bcss, tss, dunn_index])

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1],
                color='blue', label='Negative', alpha=0.6, s=10)
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1],
                color='red', label='Positive', alpha=0.6, s=10)
    plt.title(f"t-SNE (Perplexity=50, Random Seed={seed})")
    plt.legend()
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    # --- Save each image ---
    save_path = os.path.join(save_dir, f"tsne_seed_{seed}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

print("\n✅ All t-SNE images and metrics saved in folder:", save_dir)
