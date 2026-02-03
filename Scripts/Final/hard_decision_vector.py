'''import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix

# =============================
# CONFIG
# =============================
K = 3
N_FOLDS = 5
ALPHABET = {"A", "C", "G", "T"}

DATA_ROOT = "Splits"
PROB_ROOT = "output_position_kmer_probabilities_folds"
OUT_DIR = "hard_decision_results_folds"
os.makedirs(OUT_DIR, exist_ok=True)

SPECIES_LIST = ["Human", "Mouse", "Drosophila"]

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
    return sequences

# =============================
# MAJORITY LENGTH
# =============================
def majority_length(seqs):
    return Counter(len(s) for s in seqs).most_common(1)[0][0]

# =============================
# HARD DECISION VECTOR
# =============================
def hard_decision_vector(seq, pos_prob, neg_prob, target_len):
    seq = seq[:target_len]
    positions = target_len - K + 1
    votes = []

    for i in range(positions):
        kmer = seq[i:i+K]

        if len(kmer) < K or set(kmer) - ALPHABET:
            votes.append(0)
            continue

        try:
            p_pos = pos_prob.loc[kmer, f"pos_{i}"]
            p_neg = neg_prob.loc[kmer, f"pos_{i}"]
            votes.append(1 if p_pos > p_neg else 0)
        except KeyError:
            votes.append(0)

    return votes

# =============================
# MAIN EVALUATION
# =============================
def main():
    summary_rows = []

    for species in SPECIES_LIST:
        print(f"\n🧬 Species: {species}")
        species_acc = []

        for fold in range(N_FOLDS):
            print(f"  📁 Fold {fold}")

            # -----------------------------
            # Load probability matrices
            # -----------------------------
            prob_dir = f"{PROB_ROOT}/{species}/fold{fold}"

            pos_prob = pd.read_csv(
                f"{prob_dir}/{species}_pos_overlap_prob.csv",
                index_col=0
            )
            neg_prob = pd.read_csv(
                f"{prob_dir}/{species}_neg_overlap_prob.csv",
                index_col=0
            )

            # -----------------------------
            # Load test sequences
            # -----------------------------
            fold_dir = f"{DATA_ROOT}/{species}/fold{fold}"

            test_pos = read_fasta_or_txt(f"{fold_dir}/test_pos.txt")
            test_neg = read_fasta_or_txt(f"{fold_dir}/test_neg.txt")

            # -----------------------------
            # Target length (same rule)
            # -----------------------------
            if species == "Drosophila":
                target_len = majority_length(
                    read_fasta_or_txt(f"{fold_dir}/train_neg.txt")
                )
            else:
                target_len = majority_length(
                    read_fasta_or_txt(f"{fold_dir}/train_pos.txt")
                )

            y_true = []
            y_pred = []

            # -----------------------------
            # POS samples
            # -----------------------------
            for seq in test_pos:
                if len(seq) < target_len:
                    continue

                hv = hard_decision_vector(seq, pos_prob, neg_prob, target_len)
                pred = 1 if sum(hv) > 0 else 0

                y_true.append(1)
                y_pred.append(pred)

            # -----------------------------
            # NEG samples
            # -----------------------------
            for seq in test_neg:
                if len(seq) < target_len:
                    continue

                hv = hard_decision_vector(seq, pos_prob, neg_prob, target_len)
                pred = 1 if sum(hv) > 0 else 0

                y_true.append(0)
                y_pred.append(pred)

            # -----------------------------
            # Metrics
            # -----------------------------
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            species_acc.append(acc)

            print(f"    ✅ Accuracy: {acc:.4f}")
            print(f"    Confusion Matrix:\n{cm}")

            summary_rows.append({
                "species": species,
                "fold": fold,
                "accuracy": acc,
                "TP": cm[1,1],
                "FP": cm[0,1],
                "TN": cm[0,0],
                "FN": cm[1,0]
            })

        print(
            f"📊 {species} Mean Accuracy: "
            f"{np.mean(species_acc):.4f} ± {np.std(species_acc):.4f}"
        )

    # -----------------------------
    # Save summary
    # -----------------------------
    df = pd.DataFrame(summary_rows)
    df.to_csv(f"{OUT_DIR}/hard_decision_accuracy_summary.csv", index=False)

    print("\n✅ HARD DECISION EVALUATION COMPLETE")

# =============================
if __name__ == "__main__":
    main()
'''
'''import itertools
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix

# =============================
# CONFIG
# =============================
K = 3
ALPHABET = ["A", "C", "G", "T"]
FOLDS = [0, 1, 2, 3, 4]
SPECIES = ["Human", "Mouse", "Drosophila"]

BASE_DIR = "Splits"
OUT_DIR = "output_hdv_global_probability"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# FASTA / TXT READER
# =============================
def read_fasta_or_txt(filepath):
    seqs = []
    seq = ""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    seqs.append(seq.upper())
                    seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq.upper())
    return seqs

# =============================
# MAJORITY LENGTH
# =============================
def majority_length(seqs):
    return Counter(len(s) for s in seqs).most_common(1)[0][0]

# =============================
# KMERS
# =============================
KMERS = ["".join(p) for p in itertools.product(ALPHABET, repeat=K)]
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# POSITION-SPECIFIC KMER COUNTS
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
            if kmer in KMER_TO_IDX:
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat

# =============================
# COLUMN-WISE NORMALIZATION
# =============================
def column_wise_normalize(mat):
    # Laplace smoothing
    return (mat + 1) / (mat.sum(axis=1, keepdims=True) + len(KMERS))

# =============================
# HARD DECISION VECTOR
# =============================
def hard_decision_vector(seq, pos_prob, neg_prob, decision_len):
    seq = seq[:decision_len]
    positions = decision_len - K + 1
    vec = np.zeros(positions, dtype=int)

    for i in range(positions):
        kmer = seq[i:i + K]
        if kmer not in KMER_TO_IDX:
            continue

        idx = KMER_TO_IDX[kmer]
        if pos_prob[i, idx] > neg_prob[i, idx]:
            vec[i] = 1

    return vec

# =============================
# MAIN PIPELINE
# =============================
def main():

    summary = []

    for species in SPECIES:
        print(f"\n🧬 Processing species: {species}")

        # ======================================
        # 1️⃣ BUILD GLOBAL PROBABILITY (ONCE)
        # ======================================
        all_pos, all_neg = [], []

        for fold in FOLDS:
            base = f"{BASE_DIR}/{species}/fold{fold}"
            all_pos += read_fasta_or_txt(f"{base}/train_pos.txt")
            all_pos += read_fasta_or_txt(f"{base}/test_pos.txt")
            all_neg += read_fasta_or_txt(f"{base}/train_neg.txt")
            all_neg += read_fasta_or_txt(f"{base}/test_neg.txt")

        # Global probability length
        maj_len_pos = majority_length(all_pos)
        maj_len_neg = majority_length(all_neg)
        GLOBAL_PROB_LEN = min(maj_len_pos, maj_len_neg)

        print(f"  ▶ Global probability length: {GLOBAL_PROB_LEN}")

        all_pos_prob = [s[:GLOBAL_PROB_LEN] for s in all_pos if len(s) >= GLOBAL_PROB_LEN]
        all_neg_prob = [s[:GLOBAL_PROB_LEN] for s in all_neg if len(s) >= GLOBAL_PROB_LEN]

        pos_mat = kmer_freq_overlap(all_pos_prob, GLOBAL_PROB_LEN)
        neg_mat = kmer_freq_overlap(all_neg_prob, GLOBAL_PROB_LEN)

        pos_prob = column_wise_normalize(pos_mat)
        neg_prob = column_wise_normalize(neg_mat)

        # Save global probability matrices (optional but good practice)
        np.savetxt(f"{OUT_DIR}/{species}_pos_global_prob.csv", pos_prob, delimiter=",")
        np.savetxt(f"{OUT_DIR}/{species}_neg_global_prob.csv", neg_prob, delimiter=",")

        # ======================================
        # 2️⃣ FOLD-WISE EVALUATION
        # ======================================
        species_acc = []

        for fold in FOLDS:
            print(f"  Fold {fold}")

            base = f"{BASE_DIR}/{species}/fold{fold}"

            test_pos = read_fasta_or_txt(f"{base}/test_pos.txt")
            test_neg = read_fasta_or_txt(f"{base}/test_neg.txt")

            # HDV length = smallest sequence in THIS FOLD
            min_len = min(
                min(len(s) for s in test_pos),
                min(len(s) for s in test_neg)
            )

            y_true, y_pred = [], []

            for seq in test_pos:
                v = hard_decision_vector(seq, pos_prob, neg_prob, min_len)
                y_true.append(1)
                y_pred.append(1 if v.mean() >= 0.5 else 0)

            for seq in test_neg:
                v = hard_decision_vector(seq, pos_prob, neg_prob, min_len)
                y_true.append(0)
                y_pred.append(1 if v.mean() >= 0.5 else 0)

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            print(f"    Accuracy: {acc:.4f}")
            print(f"    Confusion Matrix:\n{cm}")

            summary.append({
                "species": species,
                "fold": fold,
                "accuracy": acc
            })

            species_acc.append(acc)

        print(f"  ▶ Mean accuracy ({species}): {np.mean(species_acc):.4f}")

    # ======================================
    # SAVE FINAL SUMMARY
    # ======================================
    df = pd.DataFrame(summary)
    df.to_csv(f"{OUT_DIR}/hdv_global_probability_results.csv", index=False)

    print("\n✅ GLOBAL HDV EVALUATION COMPLETED SUCCESSFULLY")

# =============================
if __name__ == "__main__":
    main()
'''
import os
import numpy as np
import pandas as pd

# ===============================
# CONFIG
# ===============================
K = 3
DATA_ROOT = "output_position_kmer_probabilities"
SPECIES = ["Human", "Mouse", "Drosophila"]
OUTPUT_DIR = "hdv_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# UTILS
# ===============================
def read_sequences(path):
    seqs = []
    with open(path) as f:
        for line in f:
            line = line.strip().upper()
            if line:
                seqs.append(line)
    return seqs


def read_global_prob(csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index("kmer")
    return df["prob"].to_dict()


# ===============================
# HARD DECISION VECTOR
# ===============================
def hard_decision_vector(seq, pos_prob, neg_prob, k):
    L = len(seq)
    positions = L - k + 1
    vec = np.zeros(positions, dtype=np.float32)

    for i in range(positions):
        kmer = seq[i:i+k]

        p_pos = pos_prob.get(kmer, 0.0)
        p_neg = neg_prob.get(kmer, 0.0)

        if p_pos > p_neg:
            vec[i] = 1.0

    return vec


# ===============================
# MAIN
# ===============================
def main():
    for sp in SPECIES:
        print(f"\nProcessing species: {sp}")

        sp_dir = os.path.join(DATA_ROOT)

        pos_csv = os.path.join(sp_dir, f"{sp}_pos_overlap_prob.csv")
        neg_csv = os.path.join(sp_dir, f"{sp}_neg_overlap_prob.csv")
        seq_file = os.path.join(sp_dir, ".txt")

        pos_prob = read_global_prob(pos_csv)
        neg_prob = read_global_prob(neg_csv)
        sequences = read_sequences(seq_file)

        hdv_all = []
        for seq in sequences:
            hdv = hard_decision_vector(seq, pos_prob, neg_prob, K)
            hdv_all.append(hdv)

        hdv_all = np.array(hdv_all, dtype=object)

        out_path = os.path.join(OUTPUT_DIR, f"{sp}_hdv.npy")
        np.save(out_path, hdv_all)

        print(f"Saved: {out_path}")
        print("Example unique values:", np.unique(hdv_all[0]))


if __name__ == "__main__":
    main()
