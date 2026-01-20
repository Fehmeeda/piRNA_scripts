# ======================================================
# Position-specific k-mer heatmap (combined Pos | Neg)
# Each k-mer is its own group
# Log normalization + row clustering
# Per species
# ======================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================
# CONFIG
# =====================
K = 3
BIN_SIZE = 5
SPECIES = ["Human", "Mouse", "Drosophila"]

DATA_DIR = "Datasets"
OUT_DIR = "Heatmaps"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# FASTA READER
# =====================
def read_fasta_sequences(filepath):
    sequences = []
    seq = ""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.upper()
        if seq:
            sequences.append(seq)
    return sequences

# =====================
# POSITION BINS
# =====================
def get_bins(seq_len, bin_size):
    return [(i, min(i + bin_size, seq_len))
            for i in range(0, seq_len, bin_size)]

# =====================
# POSITION-SPECIFIC K-MER COUNTS
# =====================
def position_kmer_counts(sequences, k, bin_size):
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for seq in sequences:
        bins = get_bins(len(seq), bin_size)
        for b_idx, (start, end) in enumerate(bins):
            for i in range(start, end - k + 1):
                kmer = seq[i:i+k]
                if len(kmer) == k and set(kmer) <= {"A","C","G","T"}:
                    counts[kmer][b_idx] += 1
                    totals[b_idx] += 1

    return counts, totals

# =====================
# COUNTS → FREQUENCY DF
# =====================
def counts_to_df(counts, totals):
    kmers = sorted(counts.keys())
    bins = sorted(totals.keys())

    data = []
    for kmer in kmers:
        row = []
        for b in bins:
            if totals[b] > 0:
                row.append(counts[kmer][b] / totals[b])
            else:
                row.append(0.0)
        data.append(row)

    columns = [f"bin_{b}" for b in bins]
    return pd.DataFrame(data, index=kmers, columns=columns)

# =====================
# LOG NORMALIZATION
# =====================
def log_normalize(df):
    return np.log1p(df)

# =====================
# COMBINE POS | NEG
# =====================
def build_combined(pos_df, neg_df):
    pos_df = pos_df.copy()
    neg_df = neg_df.copy()

    pos_df.columns = [f"Pos_{c}" for c in pos_df.columns]
    neg_df.columns = [f"Neg_{c}" for c in neg_df.columns]

    return pd.concat([pos_df, neg_df], axis=1)

# =====================
# PLOT HEATMAP
# =====================
def plot_heatmap(df, split_col, title, outpath):
    col_colors = (
        ["#4C72B0"] * split_col +
        ["#DD8452"] * (df.shape[1] - split_col)
    )

    g = sns.clustermap(
        df,
        row_cluster=True,
        col_cluster=False,
        cmap="vlag",
        col_colors=col_colors,
        figsize=(14, 10)
    )

    plt.axvline(x=split_col, color="black", linewidth=2)
    plt.title(title, y=1.05)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# =====================
# MAIN PIPELINE
# =====================
for species in SPECIES:
    print(f"Processing {species}...")

    pos_file = os.path.join(DATA_DIR, f"{species}_posi_samples.txt")
    neg_file = os.path.join(DATA_DIR, f"{species}_nega_samples.txt")

    pos_seqs = read_fasta_sequences(pos_file)
    neg_seqs = read_fasta_sequences(neg_file)

    pos_counts, pos_totals = position_kmer_counts(pos_seqs, K, BIN_SIZE)
    neg_counts, neg_totals = position_kmer_counts(neg_seqs, K, BIN_SIZE)

    df_pos = counts_to_df(pos_counts, pos_totals)
    df_neg = counts_to_df(neg_counts, neg_totals)

    df_pos = log_normalize(df_pos)
    df_neg = log_normalize(df_neg)

    df_combined = build_combined(df_pos, df_neg)

    outpath = os.path.join(
        OUT_DIR, f"{species}_position_specific_kmer_heatmap.png"
    )

    plot_heatmap(
        df_combined,
        split_col=df_pos.shape[1],
        title=f"{species}: Position-specific k-mer enrichment",
        outpath=outpath
    )

    print(f"Saved: {outpath}")

print("All species done.")
