import numpy as np
import pandas as pd
from itertools import product
from statistics import mode
from pirna import read_fasta_txt
import os

# =========================
# CONFIG
# =========================
K = 3
KMERTYPE = "disjoint"   # overlap | disjoint
DECISION_TYPE = "soft" # soft | hard
SPECIES = ["Human","Mouse","Drosophila"]
FOLDS = range(5)
OUTDIR = "output_position_kmer_probabilities"
os.makedirs(OUTDIR, exist_ok=True)

ALPHABET = "ACGT"

# =========================
# KMER UTILS
# =========================
def generate_kmers(k):
    return ["".join(p) for p in product(ALPHABET, repeat=k)]

def get_kmers(seq, k, kmertype):
    if kmertype == "overlap":
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]
    else:
        return [seq[i:i+k] for i in range(0, len(seq) - k + 1, k)]

# =========================
# MAJORITY LENGTH
# =========================
def majority_length(seqs):
    lengths = [len(s) for s in seqs]
    return mode(lengths)

# =========================
# BUILD PROB MATRIX
# =========================
def build_prob_matrix(seqs, k, kmertype, decision_type, L):
    """
    L = fixed majority length across all folds for this species
    """
    kmers = generate_kmers(k)
    kmer_to_idx = {k:i for i,k in enumerate(kmers)}

    step = 1 if kmertype == "overlap" else k
    positions = (L - k) // step + 1

    counts = np.zeros((len(kmers), positions), dtype=np.float32)

    for seq in seqs:
        if len(seq) < L:
            pad_val = 0.0 if decision_type == "soft" else 0.5
            pad_len = L - len(seq)
            seq = seq + ("N" * pad_len)
        else:
            seq = seq[:L]

        km_list = get_kmers(seq, k, kmertype)

        for pos, kmer in enumerate(km_list):
            if pos >= positions:
                break
            if "N" in kmer:
                continue
            counts[kmer_to_idx[kmer], pos] += 1

    # Column-wise normalization (P(kmer | position))
    col_sums = counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    probs = counts / col_sums

    # Row-wise normalization (optional)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    probs = probs / row_sums

    df = pd.DataFrame(
        probs,
        index=kmers,
        columns=[f"pos_{i}" for i in range(positions)]
    )
    return df

# =========================
# MAIN
# =========================
for species in SPECIES:
    print(f"\nProcessing {species}")

    # ===== COLLECT ALL SEQUENCES ACROSS ALL FOLDS =====
    all_seqs = []

    for fold in FOLDS:
        base = f"Splits/{species}/fold{fold}"
        pos = read_fasta_txt(f"{base}/train_pos.txt")
        neg = read_fasta_txt(f"{base}/train_neg.txt")
        all_seqs.extend(pos.values())
        all_seqs.extend(neg.values())

    # 🔑 Majority length across all folds, both pos & neg
    MAJ_LENGTH = majority_length(all_seqs)
    print(f"  Majority length for {species} = {MAJ_LENGTH}")

    # ===== BUILD PROB MATRIX FOR EACH FOLD =====
    for fold in FOLDS:
        base = f"Splits/{species}/fold{fold}"
        pos = read_fasta_txt(f"{base}/train_pos.txt")
        neg = read_fasta_txt(f"{base}/train_neg.txt")

        pos_df = build_prob_matrix(list(pos.values()), K, KMERTYPE, DECISION_TYPE, MAJ_LENGTH)
        neg_df = build_prob_matrix(list(neg.values()), K, KMERTYPE, DECISION_TYPE, MAJ_LENGTH)

        pos_out = f"{OUTDIR}/{species}_fold{fold}_pos_{KMERTYPE}_{DECISION_TYPE}_prob.csv"
        neg_out = f"{OUTDIR}/{species}_fold{fold}_neg_{KMERTYPE}_{DECISION_TYPE}_prob.csv"

        pos_df.to_csv(pos_out)
        neg_df.to_csv(neg_out)

        print(f"    Saved: {os.path.basename(pos_out)}")
        print(f"    Saved: {os.path.basename(neg_out)}")