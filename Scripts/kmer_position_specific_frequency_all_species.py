import itertools
import numpy as np
import plotly.graph_objects as go
from collections import Counter
import os
import pandas as pd

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

OUT_DIR = "output_position_kmer_frequency"
os.makedirs(OUT_DIR, exist_ok=True)

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
# MAJORITY LENGTH
# =============================
def majority_length(sequences):
    return Counter(len(s) for s in sequences).most_common(1)[0][0]

# =============================
# KMER LIST
# =============================
def generate_kmers(k):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

KMERS = generate_kmers(K)
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# OVERLAPPING FREQUENCY MATRIX
# =============================
'''def kmer_freq_overlap(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) != target_len:
            continue
        used += 1
        for i in range(positions):
            kmer = seq[i:i+K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat, used'''
def kmer_freq_overlap(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) < target_len:
            continue   # ❌ skip shorter sequences

        seq = seq[:target_len]  # ✅ trim longer sequences
        used += 1

        for i in range(positions):
            kmer = seq[i:i+K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat, used


# =============================
# DISJOINT FREQUENCY MATRIX
# =============================
'''def kmer_freq_disjoint(sequences, target_len):
    positions = target_len // K
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) != target_len:
            continue
        used += 1
        for i in range(positions):
            kmer = seq[i*K:(i+1)*K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1
    print(used)
    return mat, used'''
def kmer_freq_disjoint(sequences, target_len):
    positions = target_len // K
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) < target_len:
            continue   # ❌ skip shorter sequences

        seq = seq[:target_len]  # ✅ trim longer sequences
        used += 1

        for i in range(positions):
            kmer = seq[i*K:(i+1)*K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1
    print(used)

    return mat, used


# =============================
# PLOT STACKED HISTOGRAM
# =============================
def plot_histogram(matrix, title, outfile):
    fig = go.Figure()

    for i, kmer in enumerate(KMERS):
        fig.add_bar(
            x=list(range(matrix.shape[0])),
            y=matrix[:, i],
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
        height=700,
        legend_title="3-mers"
    )

    fig.write_html(outfile)
    fig.show()
    print(f"✅ Saved: {outfile}")

def save_kmer_position_table(matrix, kmers, outfile):
    """
    Convert (positions x kmers) matrix
    → (kmers x positions) table and save as CSV.
    """

    # Transpose so rows = kmers, columns = positions
    table = matrix.T  

    # Create dataframe
    df = pd.DataFrame(
        table,
        index=kmers,                                  # row names = k-mers
        columns=[f"pos_{i}" for i in range(table.shape[1])]
    )

    # Save to file
    df.to_csv(outfile)

    print(f"✅ Table saved: {outfile}")
    return df

# =============================
# MAIN
# =============================
def main():
    for species, (pos_file, neg_file) in SPECIES.items():
        print(f"\n🧬 Processing {species}")

        pos_seqs = read_fasta_or_txt(pos_file)
        neg_seqs = read_fasta_or_txt(neg_file)

        pos_len = majority_length(pos_seqs)
        neg_len = majority_length(neg_seqs)
        if species == "Drosophila":
            print("  ⚠️ Aligning Drosophila POS length to NEG majority")

            target_len = neg_len   # use NEGATIVE as reference (20)

            # Filter positives: keep only >= target_len, then trim
            pos_seqs = [s[:target_len] for s in pos_seqs if len(s) >= target_len]

            pos_len = target_len   # force both to 20
            neg_len = target_len

        print(f"  POS majority length: {pos_len}")
        print(f"  NEG majority length: {neg_len}")

        # ---------- OVERLAPPING ----------
        pos_ov, pos_used = kmer_freq_overlap(pos_seqs, pos_len)
        neg_ov, neg_used = kmer_freq_overlap(neg_seqs, neg_len)
        # ----- SAVE TABLES (OVERLAPPING) -----
        pos_df = save_kmer_position_table(
            pos_ov,
            KMERS,
            f"{OUT_DIR}/{species}_pos_overlap_table.csv"
        )

        neg_df = save_kmer_position_table(
            neg_ov,
            KMERS,
            f"{OUT_DIR}/{species}_neg_overlap_table.csv"
        )


        plot_histogram(
            pos_ov,
            f"{species} POS Overlapping {K}-mer Frequency (L={pos_len})",
            f"{OUT_DIR}/{species}_pos_overlap_freq.html"
        )

        plot_histogram(
            neg_ov,
            f"{species} NEG Overlapping {K}-mer Frequency (L={neg_len})",
            f"{OUT_DIR}/{species}_neg_overlap_freq.html"
        )

        # ---------- DISJOINT ----------
        pos_dis, _ = kmer_freq_disjoint(pos_seqs, pos_len)
        neg_dis, _ = kmer_freq_disjoint(neg_seqs, neg_len)
        # ----- SAVE TABLES (DISJOINT) -----
        save_kmer_position_table(
            pos_dis,
            KMERS,
            f"{OUT_DIR}/{species}_pos_disjoint_table.csv"
        )

        save_kmer_position_table(
            neg_dis,
            KMERS,
            f"{OUT_DIR}/{species}_neg_disjoint_table.csv"
        )


        plot_histogram(
            pos_dis,
            f"{species} POS Disjoint {K}-mer Frequency (L={pos_len})",
            f"{OUT_DIR}/{species}_pos_disjoint_freq.html"
        )

        plot_histogram(
            neg_dis,
            f"{species} NEG Disjoint {K}-mer Frequency (L={neg_len})",
            f"{OUT_DIR}/{species}_neg_disjoint_freq.html"
        )

    print("\n✔ All species processed successfully")

# =============================
if __name__ == "__main__":
    main()
