import numpy as np
from itertools import product
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pirna import read_fasta_txt, pad_sequences, get_kmers

# ============================================================
# 1. Build k-mer index mapping (DNA2VEC uses NULL=0, AAA=1..TTT=64)
# ============================================================
bases = ["A", "C", "G", "T"]
all_kmers = ["".join(p) for p in product(bases, repeat=3)]

kmer_to_index = {"NULL": 0}
for i, k in enumerate(all_kmers, start=1):
    kmer_to_index[k] = i

# ============================================================
# 2. RMS function
# ============================================================
def rms(vector):
    """Compute RMS of a single 100-d dna2vec embedding vector."""
    return np.sqrt(np.mean(vector ** 2))

# ============================================================
# 3. Compute weighted RMS per k-mer
# ============================================================
def compute_weighted_rms_per_kmer(padded_seqs, y, embedding_matrix):
    """
    For each k-mer:
        - count occurrences in positive and negative sequences
        - multiply embedding RMS by occurrence count to get weighted RMS
    """
    pos_counts = Counter()
    neg_counts = Counter()

    # Count k-mer occurrences
    for idx, seq_text in enumerate(padded_seqs.values()):
        kmers = get_kmers(seq_text, 3)
        label = y[idx]
        if label == 1:
            pos_counts.update(kmers)
        else:
            neg_counts.update(kmers)

    pos_rms = []
    neg_rms = []

    # Compute weighted RMS
    for k in all_kmers:
        emb_idx = kmer_to_index[k]
        emb_vector = embedding_matrix[emb_idx]
        rms_value = rms(emb_vector)

        # Weighted RMS = RMS * number of occurrences
        pos_rms.append(rms_value * pos_counts[k])
        neg_rms.append(rms_value * neg_counts[k])

    return pos_rms, neg_rms

# ============================================================
# 4. Heatmap
# ============================================================
def plot_rms_heatmap(pos_rms, neg_rms, species):
    data = np.array([pos_rms, neg_rms])
    plt.figure(figsize=(20,4))
    sns.heatmap(data,
                cmap="coolwarm",
                xticklabels=all_kmers,
                yticklabels=["Positive Weighted RMS", "Negative Weighted RMS"])
    plt.title(f"{species}: Weighted RMS per k-mer (dna2vec)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. MAIN PROCESSING FUNCTION
# ============================================================
def analyze_species(species_name, npz_file, pos_file, neg_file):
    print(f"\n=== Processing {species_name} ===")

    # Load label file
    data = np.load(npz_file)
    y = data["y"]  # labels

    # Load dna2vec embedding matrix
    emb_matrix = np.load("all_3mer_embeddings_with_null.npz")["embeddings"]

    # Load sequences
    all_seqs = {**read_fasta_txt(pos_file), **read_fasta_txt(neg_file)}
    padded_seqs, _ = pad_sequences(all_seqs)

    # Compute weighted RMS per k-mer
    pos_rms, neg_rms = compute_weighted_rms_per_kmer(
        padded_seqs, y, emb_matrix
    )

    # Save CSV
    df = pd.DataFrame({
        "kmer": all_kmers,
        "pos_weighted_rms": pos_rms,
        "neg_weighted_rms": neg_rms,
        "difference(pos-neg)": np.array(pos_rms) - np.array(neg_rms)
    })
    out_csv = f"{species_name}_kmer_weighted_rms.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot heatmap
    plot_rms_heatmap(pos_rms, neg_rms, species_name)

# ============================================================
# RUN ALL SPECIES
# ============================================================
analyze_species("Human",
                "Human/Human_dna2vec_embeddings.npz",
                "Datasets/Human_posi_samples.txt",
                "Datasets/Human_nega_samples.txt")

analyze_species("Mouse",
                "Mouse/Mouse_dna2vec_embeddings.npz",
                "Datasets/Mouse_posi_samples.txt",
                "Datasets/Mouse_nega_samples.txt")

analyze_species("Drosophila",
                "Drosophila/Drosophila_dna2vec_embeddings.npz",
                "Datasets/Drosophila_posi_samples.txt",
                "Datasets/Drosophila_nega_samples.txt")
