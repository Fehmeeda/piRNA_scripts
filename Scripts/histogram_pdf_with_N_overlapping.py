import os
from collections import Counter
import matplotlib.pyplot as plt
from itertools import product

# =======================================================
# 1. Generate all canonical 3-mers in order AAA → TTT
# =======================================================
def generate_all_kmers(k=3):
    alphabet = ["A", "C", "G", "T", "N"]
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    
    valid_kmers = []
    for kmer in all_kmers:
        # Skip if first character is N
        if kmer[0] == "N":
            continue
        # Skip if first two characters are NN
        if kmer[:2] == "NN":
            continue
        # Skip if middle character is N (only for k>=3)
        if kmer[1] == "N" and kmer[2] != "N" and kmer[0]!="N":
            continue
        # Skip if fully padded
        if all(c == "N" for c in kmer):
            continue
        valid_kmers.append(kmer)
    
    return valid_kmers

# =======================================================
# 2. Read FASTA sequences
# =======================================================
def load_fasta_sequences(filepath):
    seqs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().upper()
            if not line or line.startswith(">"):
                continue
            seqs.append(line)
    return seqs

# =======================================================
# 3. Pad sequences to max length
# =======================================================
def pad_sequences(sequences):
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + "N"*(max_len - len(seq)) for seq in sequences]
    return padded

# =======================================================
# 4. Extract overlapping k-mers, keep at least one real nucleotide
# =======================================================
def extract_overlapping_kmers(seq, k=3):
    kmers = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]

        # skip fully padded k-mers
        if all(c == "N" for c in kmer):
            continue
        # keep if at least one real nucleotide
        if any(c in {"A","C","G","T"} for c in kmer):
            kmers.append(kmer)
    return kmers

# =======================================================
# 5. Compute PDF in fixed k-mer order
# =======================================================
def compute_pdf(sequences, all_kmers, k=3):
    counts = Counter()
    for seq in sequences:
        kmers = extract_overlapping_kmers(seq, k)
        counts.update(kmers)

    total = sum(counts.values())
    if total == 0:
        return [0] * len(all_kmers)

    pdf = [counts[k] / total for k in all_kmers]
    return pdf

# =======================================================
# 6. Plot side-by-side histogram
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
    plt.title(f"{species}: Overlapping 3-mer PDF (Pos vs Neg, padded sequences)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# =======================================================
# 7. Process species
# =======================================================
def process_species_overlapping_with_padding(species, pos_file, neg_file, data_dir="Datasets"):
    print(f"\nProcessing {species}...")

    pos_seqs = load_fasta_sequences(os.path.join(data_dir, pos_file))
    neg_seqs = load_fasta_sequences(os.path.join(data_dir, neg_file))
    print(pos_seqs[0])

    # Pad sequences to max length across both pos & neg
    all_seqs = pos_seqs + neg_seqs
    max_len = max(len(seq) for seq in all_seqs)
    pos_seqs = [seq + "N"*(max_len - len(seq)) for seq in pos_seqs]
    neg_seqs = [seq + "N"*(max_len - len(seq)) for seq in neg_seqs]

    all_kmers = generate_all_kmers(k=3)

    pdf_pos = compute_pdf(pos_seqs, all_kmers)
    pdf_neg = compute_pdf(neg_seqs, all_kmers)

    plot_side_by_side(all_kmers, pdf_pos, pdf_neg, species)

# =======================================================
# 8. Run for all species
# =======================================================
if __name__ == "__main__":

    species_files = {
        "Human": ("Human_posi_samples.txt", "Human_nega_samples.txt"),
        "Mouse": ("Mouse_posi_samples.txt", "Mouse_nega_samples.txt"),
        "Drosophila": ("Drosophila_posi_samples.txt", "Drosophila_nega_samples.txt")
    }

    for species, (pos_file, neg_file) in species_files.items():
        process_species_overlapping_with_padding(species, pos_file, neg_file, data_dir="Datasets")
