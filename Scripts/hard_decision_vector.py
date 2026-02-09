'''import itertools
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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

OUT_DIR = "output_position_kmer_pipeline"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# FASTA / TXT READER
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
    return sequences

# =============================
# MAJORITY LENGTH
# =============================
def majority_length(seqs):
    return Counter(len(s) for s in seqs).most_common(1)[0][0]

# =============================
# KMER GENERATION
# =============================
def generate_kmers(k):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

KMERS = generate_kmers(K)
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# POSITION-SPECIFIC OVERLAPPING MATRIX
# =============================
def kmer_freq_overlap(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)
    used = 0

    for seq in sequences:
        if len(seq) < target_len:
            continue
        seq = seq[:target_len]
        used += 1

        for i in range(positions):
            kmer = seq[i:i+K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat, used

# =============================
# MATRIX → DATAFRAME
# =============================
def save_kmer_position_table(matrix, outfile):
    df = pd.DataFrame(
        matrix.T,
        index=KMERS,
        columns=[f"pos_{i}" for i in range(matrix.shape[0])]
    )
    df.to_csv(outfile)
    return df

# =============================
# NORMALIZATION
# =============================
def column_wise_normalize(df):
    return df.div(df.sum(axis=0) + 1e-12, axis=1)

def row_wise_normalize(df):
    return df.div(df.sum(axis=1) + 1e-12, axis=0)

# =============================
# HARD DECISION VECTOR
# =============================
def hard_decision_vector(seq, pos_prob, neg_prob, target_len):
    positions = target_len - K + 1
    seq = seq[:target_len]
    vec = []

    for i in range(positions):
        kmer = seq[i:i+K]
        if kmer not in pos_prob.index:
            vec.append(0)
            continue

        p_pos = pos_prob.loc[kmer, f"pos_{i}"]
        p_neg = neg_prob.loc[kmer, f"pos_{i}"]

        vec.append(1 if p_pos > p_neg else 0)

    return vec

# =============================
# MAIN PIPELINE
# =============================
def main():
    all_results = []

    for species, (pos_file, neg_file) in SPECIES.items():
        print(f"\n🧬 Processing {species}")

        pos_seqs = read_fasta_or_txt(pos_file)
        neg_seqs = read_fasta_or_txt(neg_file)

        pos_len = majority_length(pos_seqs)
        neg_len = majority_length(neg_seqs)

        # ---- Length alignment (same logic as yours) ----
        if species == "Drosophila":
            target_len = neg_len
            pos_seqs = [s[:target_len] for s in pos_seqs if len(s) >= target_len]
            neg_seqs = [s[:target_len] for s in neg_seqs if len(s) >= target_len]
        else:
            target_len = pos_len
            pos_seqs = [s[:target_len] for s in pos_seqs if len(s) >= target_len]
            neg_seqs = [s[:target_len] for s in neg_seqs if len(s) >= target_len]

        # ---- Frequency matrices ----
        pos_mat, _ = kmer_freq_overlap(pos_seqs, target_len)
        neg_mat, _ = kmer_freq_overlap(neg_seqs, target_len)

        pos_df = save_kmer_position_table(
            pos_mat, f"{OUT_DIR}/{species}_pos_overlap_raw.csv"
        )
        neg_df = save_kmer_position_table(
            neg_mat, f"{OUT_DIR}/{species}_neg_overlap_raw.csv"
        )

        # ---- Column-wise normalization ----
        pos_prob = column_wise_normalize(pos_df)
        neg_prob = column_wise_normalize(neg_df)

        pos_prob.to_csv(f"{OUT_DIR}/{species}_pos_colnorm.csv")
        neg_prob.to_csv(f"{OUT_DIR}/{species}_neg_colnorm.csv")


        # ---- Hard decision vectors ----
        for seq in pos_seqs:
            vec = hard_decision_vector(seq, pos_prob, neg_prob, target_len)
            pred = 1 if vec.count(1) > vec.count(0) else 0
            all_results.append({
                "species": species,
                "original_label": 1,
                "predicted_label": pred,
                "hard_vector": vec
            })

        for seq in neg_seqs:
            vec = hard_decision_vector(seq, pos_prob, neg_prob, target_len)
            pred = 1 if vec.count(1) > vec.count(0) else 0
            all_results.append({
                "species": species,
                "original_label": 0,
                "predicted_label": pred,
                "hard_vector": vec
            })

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(f"{OUT_DIR}/hard_decision_results.csv", index=False)
    print(final_df)
    # =============================
    # ACCURACY EVALUATION
    # =============================
    y_true = final_df["original_label"].values
    y_pred = final_df["predicted_label"].values

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n📊 HARD-DECISION PERFORMANCE")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


    print("\n✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY")

# =============================
if __name__ == "__main__":
    main()
'''

import itertools
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

OUT_DIR = "output_position_kmer_pipeline"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# FASTA / TXT READER
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
    return sequences

# =============================
# MAJORITY LENGTH
# =============================
def majority_length(seqs):
    return Counter(len(s) for s in seqs).most_common(1)[0][0]

# =============================
# KMER GENERATION
# =============================
def generate_kmers(k):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

KMERS = generate_kmers(K)
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# POSITION-SPECIFIC OVERLAPPING MATRIX
# =============================
def kmer_freq_overlap(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)

    for seq in sequences:
        if len(seq) < target_len:
            continue
        seq = seq[:target_len]

        for i in range(positions):
            kmer = seq[i:i + K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1
    

    return mat

# =============================
# MATRIX → DATAFRAME
# =============================
def save_kmer_position_table(matrix, outfile):
    df = pd.DataFrame(
        matrix.T,
        index=KMERS,
        columns=[f"pos_{i}" for i in range(matrix.shape[0])]
    )
    df.to_csv(outfile)
    return df

# =============================
# NORMALIZATION
# =============================
def column_wise_normalize(df):
    print(df.div(df.sum(axis=0) + 1e-12, axis=1))
    return df.div(df.sum(axis=0) + 1e-12, axis=1)

# =============================
# HARD DECISION VECTOR
# =============================
def hard_decision_vector(seq, pos_prob, neg_prob, decision_len):
    positions = decision_len - K + 1
    seq = seq[:decision_len]
    vec = []

    for i in range(positions):
        kmer = seq[i:i + K]
        if kmer not in pos_prob.index:
            vec.append(0)
            continue

        p_pos = pos_prob.loc[kmer, f"pos_{i}"]
        p_neg = neg_prob.loc[kmer, f"pos_{i}"]

        vec.append(1 if p_pos > p_neg else 0)

    return vec

# =============================
# MAIN PIPELINE
# =============================
def main():
    all_results = []

    for species, (pos_file, neg_file) in SPECIES.items():
        print(f"\n🧬 Processing {species}")

        pos_seqs = read_fasta_or_txt(pos_file)
        neg_seqs = read_fasta_or_txt(neg_file)

        # ---- Lengths ----
        maj_len_pos = majority_length(pos_seqs)
        maj_len_neg = majority_length(neg_seqs)
        majority_len = min(maj_len_pos, maj_len_neg)

        smallest_len = min(
            min(len(s) for s in pos_seqs),
            min(len(s) for s in neg_seqs)
        )

        print(f"📏 Majority length (probability estimation): {majority_len}")
        print(f"📏 Smallest length (hard decision): {smallest_len}")

        # ---- Truncate sequences ----
        pos_seqs_prob = [s[:majority_len] for s in pos_seqs if len(s) >= majority_len]
        neg_seqs_prob = [s[:majority_len] for s in neg_seqs if len(s) >= majority_len]

        pos_seqs_dec = [s[:smallest_len] for s in pos_seqs if len(s) >= smallest_len]
        neg_seqs_dec = [s[:smallest_len] for s in neg_seqs if len(s) >= smallest_len]

        # ---- Frequency matrices ----
        pos_mat = kmer_freq_overlap(pos_seqs_prob, majority_len)
        neg_mat = kmer_freq_overlap(neg_seqs_prob, majority_len)

        pos_df = save_kmer_position_table(
            pos_mat, f"{OUT_DIR}/{species}_pos_overlap_raw.csv"
        )
        neg_df = save_kmer_position_table(
            neg_mat, f"{OUT_DIR}/{species}_neg_overlap_raw.csv"
        )

        # ---- Column-wise normalization ----
        pos_prob = column_wise_normalize(pos_df)
        neg_prob = column_wise_normalize(neg_df)

        pos_prob.to_csv(f"{OUT_DIR}/{species}_pos_colnorm.csv")
        neg_prob.to_csv(f"{OUT_DIR}/{species}_neg_colnorm.csv")

        # ---- Hard decision vectors ----
        for seq in pos_seqs_dec:
            vec = hard_decision_vector(seq, pos_prob, neg_prob, smallest_len)
            pred = 1 if vec.count(1) > vec.count(0) else 0
            all_results.append({
                "species": species,
                "original_label": 1,
                "predicted_label": pred,
                "hard_vector": vec
            })

        for seq in neg_seqs_dec:
            vec = hard_decision_vector(seq, pos_prob, neg_prob, smallest_len)
            pred = 1 if vec.count(1) > vec.count(0) else 0
            all_results.append({
                "species": species,
                "original_label": 0,
                "predicted_label": pred,
                "hard_vector": vec
            })

    # =============================
    # FINAL RESULTS & METRICS
    # =============================
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(f"{OUT_DIR}/hard_decision_results.csv", index=False)

    print("\n📊 SPECIES-WISE PERFORMANCE")

    for species in final_df["species"].unique():
        sub = final_df[final_df["species"] == species]

        y_true = sub["original_label"].values
        y_pred = sub["predicted_label"].values

        print(f"\n🧬 {species}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("Classification Report:")
        print(classification_report(y_true, y_pred, digits=4))


    print("\n✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY")

# =============================
if __name__ == "__main__":
    main()
