# ============================================================
# CNN with predefined folds + internal validation split
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from itertools import product
from pirna import read_fasta_txt, pad_sequences
import os

# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 3
FOLDS = range(5)

SPECIES = ["Human", "Mouse", "Drosophila"]

# ============================================================
# KMER FUNCTIONS
# ============================================================
def get_overlapping_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq) - k + 1):
        kmers.append(seq[i:i+k])
    return kmers


def generate_valid_kmers(k=3, alphabet="ACGTN"):
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    valid = []

    for kmer in all_kmers:
        if kmer == "N" * k:
            valid.append(kmer)
            continue
        if kmer[0] == "N":
            continue
        if kmer[:2] == "NN":
            continue
        if k >= 3 and kmer[1] == "N" and kmer[2] != "N":
            continue
        valid.append(kmer)

    return valid


def weighted_one_hot_kmers(kmer_list, kmer_to_index, weights={0:1.0,1:0.5,2:0.25}):
    mat = np.zeros((len(kmer_to_index), len(kmer_list)), dtype=np.float32)

    for i in range(len(kmer_list)):
        for d, w in weights.items():
            if i - d >= 0:
                idx = kmer_to_index.get(kmer_list[i - d])
                if idx is not None:
                    mat[idx, i] += w
            if d != 0 and i + d < len(kmer_list):
                idx = kmer_to_index.get(kmer_list[i + d])
                if idx is not None:
                    mat[idx, i] += w
    return mat


# ============================================================
# DATA PREP
# ============================================================
'''def encode_sequences(pos_file, neg_file, kmer_to_index, max_len):
    pos = read_fasta_txt(pos_file)
    neg = read_fasta_txt(neg_file)
    all_seqs = {**pos, **neg}
    print(f"Loaded {len(pos)} positive and {len(neg)} negative sequences.")

    padded, _ = pad_sequences(all_seqs,max_len = max_len)

    X, y = [], []

    for sid, seq in padded.items():
        kmers = get_overlapping_kmers(seq, K)
        X.append(weighted_one_hot_kmers(kmers, kmer_to_index))
        y.append(1 if sid in pos else 0)

    return np.stack(X), np.array(y)'''
def encode_sequences(pos_file, neg_file, kmer_to_index, max_len):
    pos = read_fasta_txt(pos_file)
    neg = read_fasta_txt(neg_file)
    all_seqs = {**pos, **neg}

    print(f"Loaded {len(pos)} positive and {len(neg)} negative sequences.")

    X, y = [], []

    for sid, seq in all_seqs.items():
        # 🔒 FORCE padding here (no ambiguity)
        if len(seq) < max_len:
            seq = seq + "N" * (max_len - len(seq))
        else:
            seq = seq[:max_len]

        kmers = get_overlapping_kmers(seq, K)
        X.append(weighted_one_hot_kmers(kmers, kmer_to_index))
        y.append(1 if sid in pos else 0)

    return np.stack(X), np.array(y)


# ============================================================
# DATASET
# ============================================================
class KmerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# CNN MODEL
# ============================================================
class KmerCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, (5,3), padding=(2,1))
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

        with torch.no_grad():
            dummy = torch.zeros(1,1,*input_shape)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flat = x.view(1,-1).shape[1]

        self.fc1 = nn.Linear(self.flat, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_and_eval(train_X, train_y, test_X, test_y):

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_X, train_y, test_size=0.2, stratify=train_y, random_state=42
    )

    train_loader = DataLoader(KmerDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(KmerDataset(X_val, y_val), BATCH_SIZE)
    test_loader  = DataLoader(KmerDataset(test_X, test_y), BATCH_SIZE)

    model = KmerCNN(train_X.shape[1:]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    def evaluate(loader):
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                preds.extend(torch.argmax(model(xb),1).cpu().numpy())
                true.extend(yb.numpy())
        return accuracy_score(true, preds)

    return evaluate(val_loader), evaluate(test_loader)


# ============================================================
# MAIN CV LOOP
# ============================================================
if __name__ == "__main__":

    valid_kmers = generate_valid_kmers(K)
    kmer_to_index = {k:i for i,k in enumerate(valid_kmers)}
    
    SPECIES_MAX_LEN = {}

    for species in SPECIES:
        lengths = []

        for fold in FOLDS:
            base = f"Splits/{species}/fold{fold}"
        

            for file in [
                "train_pos.txt", "train_neg.txt",
                "test_pos.txt",  "test_neg.txt"
            ]:
                seqs = read_fasta_txt(f"{base}/{file}")
                lengths.extend(len(s) for s in seqs.values())

        SPECIES_MAX_LEN[species] = max(lengths)
        max_len = SPECIES_MAX_LEN[species]
        print(f"{species} max length = {SPECIES_MAX_LEN[species]}")


    for species in SPECIES:
        print(f"\n===== {species} =====")

        val_scores, test_scores = [], []

        for fold in FOLDS:
            print(f"\nFold {fold}")

            base = f"Splits/{species}/fold{fold}"
            print(species)

            X_train, y_train = encode_sequences(
                f"{base}/train_pos.txt", f"{base}/train_neg.txt", kmer_to_index, max_len
            )
            X_test, y_test = encode_sequences(
                f"{base}/test_pos.txt", f"{base}/test_neg.txt", kmer_to_index, max_len
            )

            val_acc, test_acc = train_and_eval(X_train, y_train, X_test, y_test)

            val_scores.append(val_acc)
            test_scores.append(test_acc)

            print(f"Validation Acc: {val_acc:.4f}")
            print(f"Test Acc:       {test_acc:.4f}")

        print(f"\n{species} Mean Val Acc : {np.mean(val_scores):.4f}")
        print(f"{species} Mean Test Acc: {np.mean(test_scores):.4f}")
