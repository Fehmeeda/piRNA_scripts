import os
from collections import Counter
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from scipy.interpolate import make_interp_spline


# =======================================================
# 1. Generate all canonical kmers (AAA → TTT)
# =======================================================
def generate_all_kmers(k=3):
    alphabet = ["A", "C", "G", "T"]
    return ["".join(p) for p in product(alphabet, repeat=k)]


# =======================================================
# 2. Extract VALID DISJOINT k-mers
# =======================================================
def extract_disjoint_kmers(seq, k=3):
    valid = {"A", "C", "G", "T"}
    kmers = []

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
            if not line or line.startswith(">"):
                continue
            seqs.append(line)
    return seqs


# =======================================================
# 4. Compute PMF/pdf for disjoint kmers
# =======================================================
def compute_pdf(sequences, all_kmers, k=3):
    counts = Counter()

    for seq in sequences:
        kmers = extract_disjoint_kmers(seq, k)
        counts.update(kmers)

    total = sum(counts.values())
    if total == 0:
        return [0] * len(all_kmers)

    return [counts[k] / total for k in all_kmers]


# =======================================================
# 5. Plot ONLY smooth curves (no histogram)
# =======================================================
def plot_curve_only(all_kmers, pdf_pos, pdf_neg, species):
    x = np.arange(len(all_kmers))

    # Create smooth X range
    x_smooth = np.linspace(x.min(), x.max(), 500)

    # Smooth spline curves
    spline_pos = make_interp_spline(x, pdf_pos, k=3)
    spline_neg = make_interp_spline(x, pdf_neg, k=3)

    y_pos_smooth = spline_pos(x_smooth)
    y_neg_smooth = spline_neg(x_smooth)

    plt.figure(figsize=(18, 6))

    plt.plot(x_smooth, y_pos_smooth, linewidth=2.5, label="Positive Curve")
    plt.plot(x_smooth, y_neg_smooth, linewidth=2.5, label="Negative Curve")

    plt.xticks(x, all_kmers, rotation=90)
    plt.ylabel("Probability")
    plt.title(f"{species}: Disjoint 3-mer Probability Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =======================================================
# 6. Process all species
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

        pdf_pos = compute_pdf(pos_seqs, all_kmers)
        pdf_neg = compute_pdf(neg_seqs, all_kmers)

        plot_curve_only(all_kmers, pdf_pos, pdf_neg, species)


# RUN
process_all_species("Datasets")
