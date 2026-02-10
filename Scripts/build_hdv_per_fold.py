import numpy as np
import pandas as pd
from pirna import read_fasta_txt
import os

# =========================
# CONFIG
# =========================
K = 3
KMERTYPE = "disjoint"
EPS = 1e-6
SPECIES = ["Human", "Mouse", "Drosophila"]
FOLDS = range(5)

PROB_DIR = "output_position_kmer_probabilities"
OUTDIR = "hdv_vectors"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# KMER UTILS
# =========================
def get_kmers(seq, k, kmertype):
    if kmertype == "overlap":
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]
    else:
        return [seq[i:i+k] for i in range(0, len(seq) - k + 1, k)]

# =========================
# HDV FUNCTIONS (FIXED LENGTH)
# =========================
def hard_decision_vector(seq, pos_prob, neg_prob, k, kmertype, max_pos):
    kmers = get_kmers(seq, k, kmertype)
    vec = np.zeros(max_pos, dtype=np.float32)

    for i in range(min(len(kmers), max_pos)):
        kmer = kmers[i]
        if kmer not in pos_prob.index:
            continue

        if pos_prob.loc[kmer, f"pos_{i}"] > neg_prob.loc[kmer, f"pos_{i}"]:
            vec[i] = 1.0

    return vec


def soft_decision_vector_llr(seq, pos_prob, neg_prob, k, kmertype, max_pos):
    kmers = get_kmers(seq, k, kmertype)
    vec = np.zeros(max_pos, dtype=np.float32)

    for i in range(min(len(kmers), max_pos)):
        kmer = kmers[i]
        if kmer not in pos_prob.index:
            continue

        p_pos = pos_prob.loc[kmer, f"pos_{i}"]
        p_neg = neg_prob.loc[kmer, f"pos_{i}"]

        vec[i] = np.log((p_pos + EPS) / (p_neg + EPS))

    return vec

# =========================
# MAIN
# =========================
for species in SPECIES:
    print(f"\nProcessing {species}")

    for fold in FOLDS:
        print(f"  Fold {fold}")

        # Load probability matrices
        pos_prob = pd.read_csv(
            f"{PROB_DIR}/{species}_fold{fold}_pos_{KMERTYPE}_soft_prob.csv",
            index_col=0
        )
        neg_prob = pd.read_csv(
            f"{PROB_DIR}/{species}_fold{fold}_neg_{KMERTYPE}_soft_prob.csv",
            index_col=0
        )

        # 🔑 FIXED HDV LENGTH = majority / max position length
        MAX_POS = len([c for c in pos_prob.columns if c.startswith("pos_")])

        base = f"Splits/{species}/fold{fold}"

        for split in ["train", "test"]:
            pos = read_fasta_txt(f"{base}/{split}_pos.txt")
            neg = read_fasta_txt(f"{base}/{split}_neg.txt")

            H_hard, H_soft, y, ids = [], [], [], []

            for sid, seq in {**pos, **neg}.items():
                H_hard.append(
                    hard_decision_vector(
                        seq, pos_prob, neg_prob, K, KMERTYPE, MAX_POS
                    )
                )

                H_soft.append(
                    soft_decision_vector_llr(
                        seq, pos_prob, neg_prob, K, KMERTYPE, MAX_POS
                    )
                )

                y.append(1 if sid in pos else 0)
                ids.append(sid)

            np.savez_compressed(
                f"{OUTDIR}/{species}_fold{fold}_{split}_hdv.npz",
                H_hard=np.stack(H_hard),
                H_soft=np.stack(H_soft),
                y=np.array(y),
                ids=np.array(ids)
            )

            print(f"    Saved {split} HDVs → shape = {np.stack(H_hard).shape}")