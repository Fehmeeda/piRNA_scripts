import os
import itertools
import numpy as np

# =============================
# CONFIG
# =============================
POS_FILE = "Datasets/Human_posi_samples.txt"
NEG_FILE = "Datasets/Human_nega_samples.txt"
K = 3
ALPHABET = ["A", "C", "G", "T"]

BASE_OUT = "output_state_matrices"

# =============================
# FASTA / TXT READER
# =============================
def read_fasta_or_txt(filepath):
    sequences = []
    seq = ""

    with open(filepath, "r") as f:
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

    return sequences

# =============================
# KMER DEFINITIONS
# =============================
def generate_kmers(k):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

KMERS = generate_kmers(K)
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# OVERLAPPING TRANSITION MATRIX
# =============================
def transition_matrix_overlap(seq):
    mat = np.zeros((64, 64), dtype=np.int32)

    if len(seq) < K + 1:
        return mat

    for i in range(len(seq) - K):
        kmer_from = seq[i:i+K]
        kmer_to   = seq[i+1:i+1+K]

        if set(kmer_from) <= set(ALPHABET) and set(kmer_to) <= set(ALPHABET):
            i_from = KMER_TO_IDX[kmer_from]
            i_to   = KMER_TO_IDX[kmer_to]
            mat[i_from, i_to] += 1

    return mat

# =============================
# DISJOINT TRANSITION MATRIX
# =============================
def transition_matrix_disjoint(seq):
    mat = np.zeros((64, 64), dtype=np.int32)

    if len(seq) < 2 * K:
        return mat

    steps = (len(seq) // K) - 1

    for i in range(steps):
        start1 = i * K
        start2 = (i + 1) * K

        kmer_from = seq[start1:start1+K]
        kmer_to   = seq[start2:start2+K]

        if set(kmer_from) <= set(ALPHABET) and set(kmer_to) <= set(ALPHABET):
            i_from = KMER_TO_IDX[kmer_from]
            i_to   = KMER_TO_IDX[kmer_to]
            mat[i_from, i_to] += 1
    print(mat)
    return mat

# =============================
# PROCESS & SAVE
# =============================
def process_sequences(sequences, label):
    for mode in ["overlapping", "disjoint"]:
        out_dir = os.path.join(BASE_OUT, mode, label)
        os.makedirs(out_dir, exist_ok=True)

        for idx, seq in enumerate(sequences):
            if mode == "overlapping":
                mat = transition_matrix_overlap(seq)
            else:
                mat = transition_matrix_disjoint(seq)

            out_path = os.path.join(out_dir, f"{label}_{idx}.npy")
            np.save(out_path, mat)

    print(f"✔ {label}: processed {len(sequences)} sequences")

# =============================
# MAIN
# =============================
def main():
    pos_seqs = read_fasta_or_txt(POS_FILE)
    neg_seqs = read_fasta_or_txt(NEG_FILE)

    process_sequences(pos_seqs, "positive")
    process_sequences(neg_seqs, "negative")

    print("\n✔ All variable-length transition matrices saved")
    print(f"✔ Output folder: {BASE_OUT}/")

# =============================
if __name__ == "__main__":
    main()
