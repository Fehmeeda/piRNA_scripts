# ============================================================
# 0. Imports
# ============================================================
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product
from sklearn.manifold import TSNE
from pirna import read_fasta_txt

# ============================================================
# 1. Config
# ============================================================
SPECIES = "Human"
DATA_NPZ = "Human/Human_kmer_weighted_onehot.npz"
POS_FASTA = "Datasets/Human_posi_samples.txt"
NEG_FASTA = "Datasets/Human_nega_samples.txt"

OUT_DIR = "Human/PMF_CORRECT_COMPARISONS"
os.makedirs(OUT_DIR, exist_ok=True)

X_THRESHOLD = 60
K = 3
SEED = 42
random.seed(SEED)

# ============================================================
# 2. Load features
# ============================================================
data = np.load(DATA_NPZ, allow_pickle=True)
X = data["X"]
y = data["y"]

X_flat = X.reshape(X.shape[0], -1)

# ============================================================
# 3. t-SNE
# ============================================================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca"
)
X_tsne = tsne.fit_transform(X_flat)

# ============================================================
# 4. Cluster split
# ============================================================
neg_right_idx = np.where((y == 0) & (X_tsne[:, 0] > X_THRESHOLD))[0]
neg_left_idx  = np.where((y == 0) & (X_tsne[:, 0] <= X_THRESHOLD))[0]
pos_left_idx  = np.where((y == 1) & (X_tsne[:, 0] <= X_THRESHOLD))[0]

print("Negative Right:", len(neg_right_idx))
print("Negative Left :", len(neg_left_idx))
print("Positive Left :", len(pos_left_idx))

# ============================================================
# 5. Load sequences (ORDER IS CRITICAL)
# ============================================================
pos_seqs = read_fasta_txt(POS_FASTA)
neg_seqs = read_fasta_txt(NEG_FASTA)

all_seqs = {**pos_seqs, **neg_seqs}
all_sequences = list(all_seqs.values())

def get_seqs(indices):
    return [all_sequences[i] for i in indices]

neg_right_seqs = get_seqs(neg_right_idx)
neg_left_seqs  = get_seqs(neg_left_idx)
pos_left_seqs  = get_seqs(pos_left_idx)

# ============================================================
# 6. BALANCING (key step)
# ============================================================
N = len(neg_right_seqs)

neg_left_sampled = random.sample(neg_left_seqs, N)
pos_left_sampled = random.sample(pos_left_seqs, N)

print("Balanced size:", N)

# ============================================================
# 7. k-mer utilities
# ============================================================
def generate_all_kmers(k):
    return ["".join(p) for p in product("ACGT", repeat=k)]

def extract_valid_kmers(seq, k):
    return [
        seq[i:i+k]
        for i in range(len(seq) - k + 1)
        if all(c in "ACGT" for c in seq[i:i+k])
    ]

def compute_pmf(seqs, all_kmers, k):
    counts = Counter()
    for s in seqs:
        counts.update(extract_valid_kmers(s, k))
    total = sum(counts.values())
    return np.array([counts[km] / total if total > 0 else 0 for km in all_kmers])

all_kmers = generate_all_kmers(K)

# ============================================================
# 8. PMFs
# ============================================================
pmf_neg_right = compute_pmf(neg_right_seqs, all_kmers, K)
pmf_neg_left  = compute_pmf(neg_left_sampled, all_kmers, K)
pmf_pos_left  = compute_pmf(pos_left_sampled, all_kmers, K)

# ============================================================
# 9. Plot
# ============================================================
def plot_pmf(pmf1, pmf2, label1, label2, title, path):
    x = np.arange(len(all_kmers))
    plt.figure(figsize=(18, 6))
    plt.bar(x - 0.2, pmf1, width=0.4, label=label1, alpha=0.7)
    plt.bar(x + 0.2, pmf2, width=0.4, label=label2, alpha=0.7)
    plt.xticks(x, all_kmers, rotation=90)
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

# ============================================================
# 10. CORRECT comparisons
# ============================================================

# A) Positive Left vs Negative Right
plot_pmf(
    pmf_pos_left,
    pmf_neg_right,
    "Positive-Left (sampled)",
    "Negative-Right (all)",
    f"{SPECIES}: Positive Left vs Negative Right",
    f"{OUT_DIR}/pos_left_vs_neg_right.pdf"
)

# B) Negative Left vs Negative Right
plot_pmf(
    pmf_neg_left,
    pmf_neg_right,
    "Negative-Left (sampled)",
    "Negative-Right (all)",
    f"{SPECIES}: Negative Left vs Negative Right",
    f"{OUT_DIR}/neg_left_vs_neg_right.pdf"
)

print("\n✅ Correct comparisons completed")
