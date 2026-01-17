import itertools
import numpy as np
import plotly.graph_objects as go
from collections import Counter

# =============================
# CONFIG
# =============================
POS_FILE = "Datasets/Human_posi_samples.txt"
NEG_FILE = "Datasets/Human_nega_samples.txt"
K = 3
ALPHABET = ["A", "C", "G", "T"]

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
# MAJORITY LENGTH (MODE)
# =============================
def majority_length(sequences):
    lengths = [len(s) for s in sequences]
    return Counter(lengths).most_common(1)[0][0]

# =============================
# KMER LIST
# =============================
def generate_kmers(k):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

# =============================
# OVERLAPPING KMER FREQUENCY MATRIX
# =============================
def kmer_freq_matrix_overlap(sequences, k, target_len):
    kmers = generate_kmers(k)
    kmer_to_idx = {k: i for i, k in enumerate(kmers)}

    positions = target_len - k + 1
    matrix = np.zeros((positions, len(kmers)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) != target_len:
            continue

        used += 1
        for i in range(positions):
            kmer = seq[i:i+k]
            if set(kmer) <= set(ALPHABET):
                matrix[i, kmer_to_idx[kmer]] += 1

    if used == 0:
        raise ValueError("No valid sequences for overlapping")

    return matrix, kmers, used

# =============================
# DISJOINT KMER FREQUENCY MATRIX
# =============================
def kmer_freq_matrix_disjoint(sequences, k, target_len):
    kmers = generate_kmers(k)
    kmer_to_idx = {k: i for i, k in enumerate(kmers)}

    positions = target_len // k
    matrix = np.zeros((positions, len(kmers)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) != target_len:
            continue

        used += 1
        for i in range(positions):
            start = i * k
            kmer = seq[start:start+k]
            if set(kmer) <= set(ALPHABET):
                matrix[i, kmer_to_idx[kmer]] += 1

    if used == 0:
        raise ValueError("No valid sequences for disjoint")

    return matrix, kmers, used

# =============================
# STACKED HISTOGRAM (FREQUENCY)
# =============================
def plot_kmer_histogram(matrix, kmers, title, outfile):
    fig = go.Figure()

    for idx, kmer in enumerate(kmers):
        fig.add_bar(
            x=list(range(matrix.shape[0])),
            y=matrix[:, idx],
            name=kmer,
            hovertemplate=(
                f"K-mer: {kmer}<br>"
                "Position: %{x}<br>"
                "Frequency: %{y}<extra></extra>"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Frequency",
        barmode="stack",
        height=650,
        legend_title="3-mers"
    )

    fig.write_html(outfile)
    fig.show()
    print(f"✅ Saved: {outfile}")

# =============================
# MAIN
# =============================
def main():
    pos_seqs = read_fasta_or_txt(POS_FILE)
    neg_seqs = read_fasta_or_txt(NEG_FILE)

    pos_len = majority_length(pos_seqs)
    neg_len = majority_length(neg_seqs)

    print(f"POS majority length: {pos_len}")
    print(f"NEG majority length: {neg_len}")

    # ---------- OVERLAPPING ----------
    pos_ov, kmers, pos_used = kmer_freq_matrix_overlap(pos_seqs, K, pos_len)
    neg_ov, _, neg_used = kmer_freq_matrix_overlap(neg_seqs, K, neg_len)

    print(f"POS overlapping sequences used: {pos_used}")
    print(f"NEG overlapping sequences used: {neg_used}")

    plot_kmer_histogram(
        pos_ov,
        kmers,
        f"POS Overlapping {K}-mer Frequency (L={pos_len})",
        "pos_overlap_freq.html"
    )

    plot_kmer_histogram(
        neg_ov,
        kmers,
        f"NEG Overlapping {K}-mer Frequency (L={neg_len})",
        "neg_overlap_freq.html"
    )

    # ---------- DISJOINT ----------
    pos_dis, _, pos_used = kmer_freq_matrix_disjoint(pos_seqs, K, pos_len)
    neg_dis, _, neg_used = kmer_freq_matrix_disjoint(neg_seqs, K, neg_len)

    plot_kmer_histogram(
        pos_dis,
        kmers,
        f"POS Disjoint {K}-mer Frequency (L={pos_len})",
        "pos_disjoint_freq.html"
    )

    plot_kmer_histogram(
        neg_dis,
        kmers,
        f"NEG Disjoint {K}-mer Frequency (L={neg_len})",
        "neg_disjoint_freq.html"
    )

    print("✔ All frequency plots generated successfully")

# =============================
if __name__ == "__main__":
    main()
