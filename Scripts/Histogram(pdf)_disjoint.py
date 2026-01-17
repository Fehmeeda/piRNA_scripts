#Simple disjoint (without padding) kmers proability in positive and negative samples separately

import os
from collections import Counter
import matplotlib.pyplot as plt
from itertools import product


# =======================================================
# 1. Generate all canonical 64 kmers in order AAA → TTT
# =======================================================
def generate_all_kmers(k=3):
    alphabet = ["A", "C", "G", "T"]
    return ["".join(p) for p in product(alphabet, repeat=k)]


# =======================================================
# 2. Extract VALID DISJOINT k-mers (remove N, -)
# =======================================================
def extract_disjoint_kmers(seq, k=3):
    valid = {"A", "C", "G", "T"}
    kmers = []

    # iterate in jumps of k (disjoint)
    for i in range(0, len(seq) - (len(seq) % k), k):
        kmer = seq[i:i+k]
        if all(c in valid for c in kmer):
            kmers.append(kmer)

    return kmers

# =======================================================
# 3. Load sequences
# =======================================================
def load_sequences(filepath):
    seqs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().upper()
            if not line:
                continue
            if line.startswith(">"):
                continue  # skip FASTA titles
            seqs.append(line)
    return seqs



# =======================================================
# 4. Compute PDF in fixed order of kmers
# =======================================================
def compute_pdf(sequences, all_kmers, k=3):
    counts = Counter()

    for seq in sequences:
        kmers = extract_disjoint_kmers(seq, k)
        counts.update(kmers)

    total = sum(counts.values())
    if total == 0:
        return [0] * len(all_kmers)

    pdf = [counts[k] / total for k in all_kmers]
    return pdf


# =======================================================
# 5. Plot side-by-side bars (pos vs neg)
# =======================================================
def plot_side_by_side(all_kmers, pdf_pos, pdf_neg, species):
    x = range(len(all_kmers))
    bar_width = 0.4

    plt.figure(figsize=(18, 6))

    plt.bar([xi - bar_width/2 for xi in x], pdf_pos,
            width=bar_width, label="Positive", alpha=0.8)

    plt.bar([xi + bar_width/2 for xi in x], pdf_neg,
            width=bar_width, label="Negative", alpha=0.8)

    plt.xticks(x, all_kmers, rotation=90)
    plt.ylabel("Probability")
    plt.title(f"{species}: Disjoint 3-mer PDF (Pos vs Neg)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =======================================================
# 6. Process all species using your dataset files
# =======================================================
def process_all_species(data_dir="Datasets"):
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
        print(pos_seqs[0])

        pdf_pos = compute_pdf(pos_seqs, all_kmers)
        pdf_neg = compute_pdf(neg_seqs, all_kmers)

        plot_side_by_side(all_kmers, pdf_pos, pdf_neg, species)


# RUN
process_all_species("Datasets")
