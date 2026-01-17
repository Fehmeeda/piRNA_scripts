import os
import itertools
import numpy as np

# =============================
# CONFIG
# =============================
K = 3
ALPHABET = ["A", "C", "G", "T"]

SPECIES = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila":   ("Datasets/Drosophila_posi_samples.txt",   "Datasets/Drosophila_nega_samples.txt"),
}

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

    return mat

# =============================
# PROCESS & SAVE
# =============================
def process_sequences(sequences, species, label):
    for mode in ["overlapping", "disjoint"]:
        out_dir = os.path.join(BASE_OUT, species, mode, label)
        os.makedirs(out_dir, exist_ok=True)

        for idx, seq in enumerate(sequences):
            if mode == "overlapping":
                mat = transition_matrix_overlap(seq)
            else:
                mat = transition_matrix_disjoint(seq)

            out_path = os.path.join(out_dir, f"{label}_{idx}.npy")
            np.save(out_path, mat)

    print(f"✔ {species} | {label}: {len(sequences)} sequences processed")

# =============================
# MAIN
# =============================
def main():
    for species, (pos_file, neg_file) in SPECIES.items():
        print(f"\n🧬 Processing species: {species}")

        pos_seqs = read_fasta_or_txt(pos_file)
        neg_seqs = read_fasta_or_txt(neg_file)

        process_sequences(pos_seqs, species, "positive")
        process_sequences(neg_seqs, species, "negative")

    print("\n✔ All species processed successfully")
    print(f"✔ Output root folder: {BASE_OUT}/")

# =============================
if __name__ == "__main__":
    main()
