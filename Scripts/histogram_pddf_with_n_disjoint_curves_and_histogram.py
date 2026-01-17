import os
from collections import Counter
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from scipy.interpolate import make_interp_spline


# =======================================================
# 1. Generate all canonical 64 kmers (AAA → TTT) with N handling
# =======================================================
def generate_all_kmers(k=3):
    alphabet = ["A", "C", "G", "T", "N"]
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    
    valid_kmers = []
    for kmer in all_kmers:
        if kmer[0] == "N": continue
        if kmer[:2] == "NN": continue
        if kmer[1] == "N" and kmer[2] != "N" and kmer[0] != "N": continue
        if all(c == "N" for c in kmer): continue
        valid_kmers.append(kmer)
    
    return valid_kmers


# =======================================================
# 2. Load FASTA sequences
# =======================================================
def load_fasta_sequences(filepath):
    seqs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().upper()
            if not line: continue
            if line.startswith(">"): continue
            seqs.append(line)
    return seqs


# =======================================================
# 3. Extract disjoint k-mers
# =======================================================
def extract_disjoint_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq) - (len(seq) % k), k):
        kmer = seq[i:i+k]
        if all(c == "N" for c in kmer): continue
        if any(c in {"A","C","G","T"} for c in kmer):
            kmers.append(kmer)
    return kmers


# =======================================================
# 4. Compute PDF in fixed k-mer order
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
# 5. Plot histogram + curve together
# =======================================================
def plot_bar_and_curve(all_kmers, pdf_pos, pdf_neg, species):
    x = np.arange(len(all_kmers))
    bar_width = 0.4

    plt.figure(figsize=(18, 6))

    # Bars
    plt.bar([xi - bar_width/2 for xi in x], pdf_pos,
            width=bar_width, label="Positive (bar)", alpha=0.6)
    plt.bar([xi + bar_width/2 for xi in x], pdf_neg,
            width=bar_width, label="Negative (bar)", alpha=0.6)

    # Smooth curves
    x_smooth = np.linspace(x.min(), x.max(), 500)
    spline_pos = make_interp_spline(x, pdf_pos, k=3)
    spline_neg = make_interp_spline(x, pdf_neg, k=3)
    y_pos_smooth = spline_pos(x_smooth)
    y_neg_smooth = spline_neg(x_smooth)

    plt.plot(x_smooth, y_pos_smooth, linewidth=2.5, label='Positive (curve)')
    plt.plot(x_smooth, y_neg_smooth, linewidth=2.5, label='Negative (curve)')

    plt.xticks(x, all_kmers, rotation=90)
    plt.ylabel("Probability")
    plt.title(f"{species}: Disjoint 3-mer PDF (Pos vs Neg, padded sequences)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =======================================================
# 6. Process species
# =======================================================
def process_species_disjoint_with_padding(species, pos_file, neg_file, data_dir="Datasets"):
    print(f"\nProcessing {species}...")

    pos_seqs = load_fasta_sequences(os.path.join(data_dir, pos_file))
    neg_seqs = load_fasta_sequences(os.path.join(data_dir, neg_file))

    # Pad sequences to max length across both pos & neg
    all_seqs = pos_seqs + neg_seqs
    max_len = max(len(seq) for seq in all_seqs)
    pos_seqs = [seq + "N"*(max_len - len(seq)) for seq in pos_seqs]
    neg_seqs = [seq + "N"*(max_len - len(seq)) for seq in neg_seqs]

    all_kmers = generate_all_kmers(k=3)

    pdf_pos = compute_pdf(pos_seqs, all_kmers)
    pdf_neg = compute_pdf(neg_seqs, all_kmers)

    plot_bar_and_curve(all_kmers, pdf_pos, pdf_neg, species)


# =======================================================
# 7. Run for all species
# =======================================================
if __name__ == "__main__":
    species_files = {
        "Human": ("Human_posi_samples.txt", "Human_nega_samples.txt"),
        "Mouse": ("Mouse_posi_samples.txt", "Mouse_nega_samples.txt"),
        "Drosophila": ("Drosophila_posi_samples.txt", "Drosophila_nega_samples.txt")
    }

    for species, (pos_file, neg_file) in species_files.items():
        process_species_disjoint_with_padding(species, pos_file, neg_file, data_dir="Datasets")
