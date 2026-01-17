import os
from collections import Counter
import matplotlib.pyplot as plt
from itertools import product
import numpy as np


# =======================================================
# 1. Generate all canonical kmers (AAA → TTT)
# =======================================================
def generate_all_kmers(k=3):
    alphabet = ["A", "C", "G", "T"]
    return ["".join(p) for p in product(alphabet, repeat=k)]


# =======================================================
# 2. Extract VALID DISJOINT kmers (skip N, -)
# =======================================================
def extract_valid_kmers_disjoint(seq, k=3):
    valid = {"A", "C", "G", "T"}
    kmers = []
    # i jumps by k → non-overlapping
    for i in range(0, len(seq) - k + 1, k):
        kmer = seq[i:i+k]
        if all(c in valid for c in kmer):
            kmers.append(kmer)
    return kmers


# =======================================================
# 3. Load sequences from file
# =======================================================
def load_sequences(filepath):
    seqs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().upper()
            if not line or line.startswith(">"):
                continue
            seqs.append(line)
    return seqs


# =======================================================
# 4. Compute PMF for DISJOINT kmers
# =======================================================
def compute_pmf_disjoint(sequences, all_kmers, k=3):
    counts = Counter()
    for seq in sequences:
        counts.update(extract_valid_kmers_disjoint(seq, k))

    total = sum(counts.values())
    if total == 0:
        return [0] * len(all_kmers)

    pmf = [counts[kmer] / total for kmer in all_kmers]
    return pmf


# =======================================================
# 5. Heatmap visualization of kmer PDFs
# =======================================================
def plot_heatmap(all_kmers, pdf_pos, pdf_neg, species):

    data = np.array([pdf_pos, pdf_neg])  # shape: (2, 64)

    plt.figure(figsize=(18, 4))
    plt.imshow(data, cmap="viridis", aspect="auto")

    plt.colorbar(label="Probability")

    plt.yticks([0, 1], ["Positive", "Negative"])
    plt.xticks(range(len(all_kmers)), all_kmers, rotation=90)

    plt.title(f"{species}: Disjoint 3-mer Probability Heatmap")
    plt.tight_layout()
    plt.show()


# =======================================================
# 6. Process All Species
# =======================================================
def process_all_species_disjoint(data_dir="Datasets"):
    species_files = {
        "Human": ("Human_posi_samples.txt", "Human_nega_samples.txt"),
        "Mouse": ("Mouse_posi_samples.txt", "Mouse_nega_samples.txt"),
        "Drosophila": ("Drosophila_posi_samples.txt", "Drosophila_nega_samples.txt")
    }

    all_kmers = generate_all_kmers(k=3)

    for species, (pos_file, neg_file) in species_files.items():
        print(f"\nProcessing {species}...")

        pos_seqs = load_sequences(os.path.join(data_dir, pos_file))
        neg_seqs = load_sequences(os.path.join(data_dir, neg_file))

        pmf_pos = compute_pmf_disjoint(pos_seqs, all_kmers)
        pmf_neg = compute_pmf_disjoint(neg_seqs, all_kmers)

        plot_heatmap(all_kmers, pmf_pos, pmf_neg, species)


# =======================================================
# RUN SCRIPT
# =======================================================
process_all_species_disjoint("Datasets")
