import itertools
import numpy as np
import pandas as pd
import os
from collections import Counter

# =============================
# CONFIG
# =============================
K = 3
ALPHABET = ["A", "C", "G", "T"]

SPECIES = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": (
        "Datasets/Drosophila_posi_samples.txt",
        "Datasets/Drosophila_nega_samples.txt"
    ),
}

OUT_DIR = "output_position_kmer_probabilities"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# READ FASTA / TXT
# =============================
def read_fasta_or_txt(filepath):
    sequences = []
    seq = ""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq.upper())
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq.upper())
    print(sequences[0])
    return sequences

# =============================
# MAJORITY LENGTH
# =============================
def majority_length(seqs):
    return Counter(len(s) for s in seqs).most_common(1)[0][0]
def min_length(sequences):
    return min(len(s) for s in sequences)
# =============================
# KMER SETUP
# =============================
def generate_kmers(k):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

KMERS = generate_kmers(K)
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# OVERLAPPING KMER MATRIX
# =============================
def overlapping_kmer_matrix(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) < target_len:
            continue
        used += 1
        seq = seq[:target_len]

        for i in range(positions):
            kmer = seq[i:i+K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat, used

# =============================
# DISJOINT KMER MATRIX
# =============================
def disjoint_kmer_matrix(sequences, target_len):
    positions = target_len // K
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) < positions * K:
            continue
        used += 1
        seq = seq[:positions * K]

        for i in range(positions):
            kmer = seq[i*K:(i+1)*K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat, used

# =============================
# SAVE MATRIX + PROBABILITY
# =============================
def save_matrix_and_prob(matrix, species, label, mode):
    df = pd.DataFrame(
        matrix.T,
        index=KMERS,
        columns=[f"pos_{i}" for i in range(matrix.shape[0])]
    )

    raw_path = f"{OUT_DIR}/{species}_{label}_{mode}_raw.csv"
    prob_path = f"{OUT_DIR}/{species}_{label}_{mode}_prob.csv"

    df.to_csv(raw_path)
    prob_df = df.div(df.sum(axis=0) + 1e-12, axis=1)
    prob_df.to_csv(prob_path)

# =============================
# MAIN
# =============================
def main():
    for species, (pos_file, neg_file) in SPECIES.items():
        print(f"\n🧬 Processing {species}")

        pos_seqs = read_fasta_or_txt(pos_file)
        neg_seqs = read_fasta_or_txt(neg_file)

        '''# ---- Length decision ----
        if species == "Drosophila":
            target_len = min_length(neg_seqs)
            rule = "mini NEGATIVE"
        else:'''
        target_len = min_length(pos_seqs)
        #rule = "mini POSITIVE"

        #print(f"📏 Target length: {target_len} ({rule})")

        pos_seqs = [s[:target_len] for s in pos_seqs if len(s) >= target_len]
        neg_seqs = [s[:target_len] for s in neg_seqs if len(s) >= target_len]

        print(f"✅ POS samples used: {len(pos_seqs)}")
        print(f"✅ NEG samples used: {len(neg_seqs)}")

        # ---- Overlapping ----
        pos_mat, pos_n = overlapping_kmer_matrix(pos_seqs, target_len)
        neg_mat, neg_n = overlapping_kmer_matrix(neg_seqs, target_len)

        save_matrix_and_prob(pos_mat, species, "pos", "overlap")
        save_matrix_and_prob(neg_mat, species, "neg", "overlap")

        # ---- Disjoint ----
        pos_mat_d, pos_nd = disjoint_kmer_matrix(pos_seqs, target_len)
        neg_mat_d, neg_nd = disjoint_kmer_matrix(neg_seqs, target_len)

        save_matrix_and_prob(pos_mat_d, species, "pos", "disjoint")
        save_matrix_and_prob(neg_mat_d, species, "neg", "disjoint")

        print("💾 Saved overlapping + disjoint probability matrices")

    print("\n✅ PROBABILITY ESTIMATION PIPELINE FINISHED")

# =============================
if __name__ == "__main__":
    main()
