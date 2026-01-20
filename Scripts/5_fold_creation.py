import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold


def read_fasta_txt(file_path):
    sequences = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            sequences.append(line.upper())
    return sequences


def load_sequences(pos_file, neg_file):
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    X = np.array(pos_seqs + neg_seqs)
    y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

    print(f"Loaded {len(pos_seqs)} positive samples")
    print(f"Loaded {len(neg_seqs)} negative samples")

    return X, y


def save_fold_txt(X, y, train_idx, test_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def write(fname, idx, label):
        with open(fname, "w") as f:
            for i in idx:
                if y[i] == label:
                    f.write(X[i] + "\n")

    write(os.path.join(out_dir, "train_pos.txt"), train_idx, 1)
    write(os.path.join(out_dir, "train_neg.txt"), train_idx, 0)
    write(os.path.join(out_dir, "test_pos.txt"),  test_idx, 1)
    write(os.path.join(out_dir, "test_neg.txt"),  test_idx, 0)


def create_5fold_cv(X, y, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # 1️⃣ Save indices (for reproducibility)
        fold_data = {
            "train_idx": train_idx,
            "test_idx": test_idx
        }

        with open(os.path.join(out_dir, f"fold{fold}.pkl"), "wb") as f:
            pickle.dump(fold_data, f)

        # 2️⃣ Save sequences (for models)
        fold_txt_dir = os.path.join(out_dir, f"fold{fold}")
        save_fold_txt(X, y, train_idx, test_idx, fold_txt_dir)

        print(
            f"Fold {fold}: "
            f"train={len(train_idx)} "
            f"test={len(test_idx)} "
            f"(pos={y[test_idx].sum()}, "
            f"neg={len(test_idx) - y[test_idx].sum()})"
        )

    print("\n✅ 5-fold CV splits saved as PKL + TXT in:", out_dir)


# =============================
# RUN
# =============================
pos_file = "Datasets/Human_posi_samples.txt"
neg_file = "Datasets/Human_nega_samples.txt"
out_dir  = "Splits/Human"

X, y = load_sequences(pos_file, neg_file)
create_5fold_cv(X, y, out_dir)
