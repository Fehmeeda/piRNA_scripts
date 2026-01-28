import numpy as np
from itertools import product
import os
from pirna import read_fasta_txt, pad_sequences


# ============================================================
# 1. DISJOINT KMER GENERATION
# ============================================================
def get_disjoint_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq), k):
        chunk = seq[i:i+k]
        if len(chunk) < k:
            chunk += 'N' * (k - len(chunk))
        kmers.append(chunk)
    return kmers


# ============================================================
# 2. VALID KMER GENERATOR (N-handling rules)
# ============================================================
def generate_valid_kmers(k=3, alphabet='ACGTN'):
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    valid_kmers = []

    for kmer in all_kmers:

        if kmer == 'N' * k:
            valid_kmers.append(kmer)
            continue

        if kmer[0] == 'N':
            continue

        if kmer[:2] == 'NN':
            continue

        if k >= 3 and kmer[1] == 'N' and kmer[2] != 'N' and kmer[0] != 'N':
            continue

        valid_kmers.append(kmer)

    return valid_kmers


# ============================================================
# 3. WEIGHTED ONE-HOT (DISJOINT)
# ============================================================
def weighted_one_hot_kmers(kmer_list, kmer_to_index, weight_scheme=None):

    if weight_scheme is None:
        weight_scheme = {0: 1.0, 1: 0.5, 2: 0.25}

    vec_len = len(kmer_to_index)
    seq_len = len(kmer_list)

    mat = np.zeros((vec_len, seq_len), dtype=float)

    for i, kmer in enumerate(kmer_list):

        for dist, weight in weight_scheme.items():

            if i - dist >= 0:
                idx = kmer_to_index.get(kmer_list[i - dist])
                if idx is not None:
                    mat[idx, i] += weight

            if dist != 0 and i + dist < seq_len:
                idx = kmer_to_index.get(kmer_list[i + dist])
                if idx is not None:
                    mat[idx, i] += weight

    return mat


# ============================================================
# 4. PROCESS ONE SPECIES (ONLY ONE-HOT)
# ============================================================
def process_species(species_name, pos_file, neg_file):

    print(f"\nProcessing {species_name}")

    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)
    all_seqs = {**pos_seqs, **neg_seqs}

    padded_all, max_len = pad_sequences(all_seqs)
    print(f"Padded length: {max_len}")

    valid_kmers = generate_valid_kmers(k=3)
    kmer_to_index = {kmer: i for i, kmer in enumerate(valid_kmers)}

    X_list, y_list = [], []

    for i, (seq_id, seq) in enumerate(padded_all.items()):

        kmers = get_disjoint_kmers(seq, k=3)
        weighted_oh = weighted_one_hot_kmers(kmers, kmer_to_index)

        X_list.append(weighted_oh)
        y_list.append(1 if seq_id in pos_seqs else 0)

        if i == 0:
            print("Example weighted one-hot shape:", weighted_oh.shape)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    os.makedirs(species_name, exist_ok=True)
    np.savez_compressed(
        f"{species_name}/{species_name}_kmer_weighted_disjoint_onehot.npz",
        X=X,
        y=y
    )

    print(f"Saved {species_name}_kmer_weighted_disjoint_onehot.npz")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    species = {
        "Human": ("Datasets/Human_posi_samples.txt",
                  "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt",
                  "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt",
                       "Datasets/Drosophila_nega_samples.txt"),
    }

    for sp, (pos_file, neg_file) in species.items():
        process_species(sp, pos_file, neg_file)
