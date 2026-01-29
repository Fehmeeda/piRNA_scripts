import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold

# =============================
# FASTA reader (preserve IDs)
# =============================
def read_fasta_with_ids(file_path, label):
    sequences = []
    labels = []
    ids = []

    curr_id = None

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                curr_id = line[1:]
            else:
                sequences.append(line.upper())
                labels.append(label)
                ids.append(curr_id)

    return sequences, labels, ids


# =============================
# Load dataset
# =============================
def load_sequences_with_ids(pos_file, neg_file):
    X, y, ids = [], [], []

    pos_X, pos_y, pos_ids = read_fasta_with_ids(pos_file, 1)
    neg_X, neg_y, neg_ids = read_fasta_with_ids(neg_file, 0)

    X.extend(pos_X)
    X.extend(neg_X)

    y.extend(pos_y)
    y.extend(neg_y)

    ids.extend(pos_ids)
    ids.extend(neg_ids)

    return np.array(X), np.array(y), np.array(ids)


# =============================
# Balance indices
# =============================
def balance_indices(idx, y, seed):
    pos_idx = idx[y[idx] == 1]
    neg_idx = idx[y[idx] == 0]

    n = min(len(pos_idx), len(neg_idx))

    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(pos_idx, n, replace=False)
    neg_idx = rng.choice(neg_idx, n, replace=False)

    balanced_idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(balanced_idx)

    return balanced_idx


# =============================
# Save FASTA fold
# =============================
def save_fold_fasta(X, y, ids, idx, label, out_file):
    with open(out_file, "w") as f:
        for i in idx:
            if y[i] == label:
                f.write(f">{ids[i]}\n")
                f.write(X[i] + "\n")


# =============================
# Create 5-fold CV (balanced)
# =============================
def create_5fold_cv(X, y, ids, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        # 🔒 Balance train and test separately
        train_idx = balance_indices(train_idx, y, seed + fold)
        test_idx  = balance_indices(test_idx,  y, seed + fold)

        fold_dir = os.path.join(out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save PKL (exact IDs)
        fold_data = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_ids": ids[train_idx],
            "test_ids": ids[test_idx]
        }

        with open(os.path.join(out_dir, f"fold{fold}.pkl"), "wb") as f:
            pickle.dump(fold_data, f)

        # Save FASTA files
        save_fold_fasta(X, y, ids, train_idx, 1, f"{fold_dir}/train_pos.txt")
        save_fold_fasta(X, y, ids, train_idx, 0, f"{fold_dir}/train_neg.txt")
        save_fold_fasta(X, y, ids, test_idx,  1, f"{fold_dir}/test_pos.txt")
        save_fold_fasta(X, y, ids, test_idx,  0, f"{fold_dir}/test_neg.txt")

        # Sanity check
        tr_pos = np.sum(y[train_idx] == 1)
        tr_neg = np.sum(y[train_idx] == 0)
        te_pos = np.sum(y[test_idx] == 1)
        te_neg = np.sum(y[test_idx] == 0)

        print(
            f"Fold {fold} | "
            f"Train: {tr_pos} pos / {tr_neg} neg | "
            f"Test: {te_pos} pos / {te_neg} neg"
        )

    print("\n✅ CV saved with ORIGINAL IDs and PERFECT class balance per fold")


# =============================
# RUN (species-wise)
# =============================
if __name__ == "__main__":

    pos_file = "Datasets/Human_posi_samples.txt"
    neg_file = "Datasets/Human_nega_samples.txt"
    out_dir  = "Splits/Human"

    X, y, ids = load_sequences_with_ids(pos_file, neg_file)
    create_5fold_cv(X, y, ids, out_dir)
