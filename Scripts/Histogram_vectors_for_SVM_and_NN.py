import os
import numpy as np
from collections import Counter
from itertools import product

# ============================================
# 1. Generate all canonical k-mers (AAA → TTT)
# ============================================
def generate_all_kmers(k=3):
    alphabet = ["A", "C", "G", "T"]
    return ["".join(p) for p in product(alphabet, repeat=k)]


# ============================================
# 2. Extract VALID overlapping k-mers
# ============================================
def extract_valid_kmers(seq, k=3):
    valid = {"A", "C", "G", "T"}
    kmers = []

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(c in valid for c in kmer):
            kmers.append(kmer)

    return kmers


# ============================================
# 3. Load FASTA-like sequences
# ============================================
def load_sequences(filepath):
    seqs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().upper()
            if not line or line.startswith(">"):
                continue
            seqs.append(line)
    return seqs


# ============================================
# 4. Compute k-mer PROBABILITY vector
#    (ONE sequence → ONE vector)
# ============================================
def compute_kmer_probability_vector(seq, all_kmers, k=3):
    kmers = extract_valid_kmers(seq, k)
    total = len(kmers)

    if total == 0:
        return np.zeros(len(all_kmers))

    counts = Counter(kmers)
    return np.array([counts[km] / total for km in all_kmers])


# ============================================
# 5. Build ML dataset (X, y)
# ============================================
def build_dataset(pos_seqs, neg_seqs, all_kmers, k=3):
    X = []
    y = []

    for seq in pos_seqs:
        X.append(compute_kmer_probability_vector(seq, all_kmers, k))
        y.append(1)

    for seq in neg_seqs:
        X.append(compute_kmer_probability_vector(seq, all_kmers, k))
        y.append(0)
    
    

    return np.array(X), np.array(y)


# ============================================
# 6. Process all species and SAVE features
# ============================================
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

        X, y = build_dataset(pos_seqs, neg_seqs, all_kmers, k=3)

        np.save(f"{species}_X_kmer_prob.npy", X)
        np.save(f"{species}_y.npy", y)

        print(f"Saved {species}_X_kmer_prob.npy  → shape {X.shape}")
        print(f"Saved {species}_y.npy            → shape {y.shape}")


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    process_all_species("Datasets")
