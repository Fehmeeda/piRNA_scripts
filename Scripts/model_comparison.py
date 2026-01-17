import matplotlib.pyplot as plt
import numpy as np

# =========================
# BEST RESULTS (summarized)
# =========================

data = {
    "Human": {
        "SVM-Kmer":  (0.77, 0.86),
        "SVM-tSNE":  (0.7262, 0.7785),
        "SVM-PCA":   (0.7539, 0.8247),
        "NN-Kmer":   (0.77, 0.84),
        "NN-tSNE":   (0.75, 0.8222),
        "NN-PCA":    (0.75, 0.8230),

    },

    "Mouse": {
        "SVM-Kmer":  (0.79, 0.86),
        "SVM-tSNE":  (0.7043, 0.7601),
        "SVM-PCA":   (0.6882, 0.7525),
        "NN-Kmer":   (0.79, 0.86),
        "NN-tSNE":   (0.73, 0.8065),
        "NN-PCA":    (0.68, 0.7533),
    },

    "Drosophila": {
        "SVM-Kmer":  (0.87, 0.94),
        "SVM-tSNE":  (0.9403, 0.9757),
        "SVM-PCA":   (0.9468, 0.9853),
        "NN-Kmer":   (0.83, 0.90),
        "NN-tSNE":   (0.94, 0.9799),
        "NN-PCA":    (0.95, 0.9895),
    }
}

# =========================
# PLOTTING
# =========================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

bar_width = 0.35

for ax, (species, results) in zip(axes, data.items()):
    labels = list(results.keys())
    accuracy = [v[0] for v in results.values()]
    auc = [v[1] for v in results.values()]

    x = np.arange(len(labels))

    ax.bar(x - bar_width/2, accuracy, bar_width, label="Accuracy")
    ax.bar(x + bar_width/2, auc, bar_width, label="AUC")

    ax.set_title(species)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

axes[0].set_ylabel("Performance")
axes[1].legend(loc="lower right")

plt.suptitle("Performance Comparison of Models Across Species", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
