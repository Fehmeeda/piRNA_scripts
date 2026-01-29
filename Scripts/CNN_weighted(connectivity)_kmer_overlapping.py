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
LR = 1e-4
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

#If i just remove the N, it will not include any of the Kmers with N in the combination (all kmers product)
def generate_valid_kmers(k=3, alphabet="ACGT"):
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

'''def weighted_one_hot_kmers(kmer_list, kmer_to_index, weights={0:1.0,1:0.5,2:0.25}):
    mat = np.zeros((len(kmer_to_index), len(kmer_list)), dtype=np.float32)

    for i in range(len(kmer_list)):
        kmer = kmer_list[i]
        if 'N' in kmer:
            continue  # skip k-mers with N
        for d, w in weights.items():
            if i - d >= 0:
                idx = kmer_to_index.get(kmer_list[i - d])
                if idx is not None and 'N' not in kmer_list[i - d]:
                    mat[idx, i] += w
            if d != 0 and i + d < len(kmer_list):
                idx = kmer_to_index.get(kmer_list[i + d])
                if idx is not None and 'N' not in kmer_list[i + d]:
                    mat[idx, i] += w

    
    return mat'''


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

    #print(f"Loaded {len(pos)} positive and {len(neg)} negative sequences.")

    X, y = [], []

    for sid, seq in all_seqs.items():
        # FORCE padding here 
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
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

'''def get_model_architecture(input_shape):
    model = KmerCNN(input_shape)
    return str(model)'''
def get_model_architecture(input_shape):
    num_kmers, seq_len = input_shape
    model = KmerCNN(seq_len, num_kmers)
    return str(model)

# ============================================================
# CNN MODEL
# ============================================================
'''class KmerCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, (5,3))
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

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
'''
class KmerCNN(nn.Module):
    def __init__(self, seq_len, num_kmers):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_kmers,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        self.fc = nn.Linear((seq_len // 2) * 64, 2)

    def forward(self, x):
        # x: B × num_kmers × positions
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)

# ============================================================
# TRAIN / EVAL
# ============================================================
'''def train_and_eval(train_X, train_y, test_X, test_y):

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
'''
def train_and_eval(train_X, train_y, test_X, test_y):

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_X, train_y, test_size=0.2, stratify=train_y,random_state=42
    )

    train_loader = DataLoader(KmerDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(KmerDataset(X_val, y_val), BATCH_SIZE)
    test_loader  = DataLoader(KmerDataset(test_X, test_y), BATCH_SIZE)
    
    print("Class distribution:")
    print("Train:", np.bincount(y_tr))
    print("Val:  ", np.bincount(y_val))
    print(f"Train samples: {len(y_tr)}, Val samples: {len(y_val)}, Test samples: {len(test_y)}")

    num_kmers = train_X.shape[1]
    seq_len   = train_X.shape[2]
    print(f"Number of kmers: {num_kmers}, Sequence length: {seq_len}")

    model = KmerCNN(seq_len, num_kmers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    best_val = float("inf")
    patience = 5
    counter = 0

    for epoch in range(EPOCHS):

        # ===== TRAIN =====
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += crit(model(xb), yb).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # ===== EARLY STOPPING =====
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), f"{species}_model.pt")
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(f"{species}_model.pt"))




    # ===== METRIC EVALUATION =====
    def evaluate(loader):
        model.eval()
        probs, preds, true = [], [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                p = torch.softmax(out, dim=1)[:,1].cpu().numpy()

                probs.extend(p)
                preds.extend((p >= 0.5).astype(int))
                true.extend(yb.numpy())

        return {
            "acc": accuracy_score(true, preds),
            "auc": roc_auc_score(true, probs),
            "precision": precision_score(true, preds),
            "recall": recall_score(true, preds),
            "f1": f1_score(true, preds)
        }

    return (
        evaluate(val_loader),
        evaluate(test_loader),
        train_losses,
        val_losses
    )
RESULTS_DIR = "results_cnn"
os.makedirs(RESULTS_DIR, exist_ok=True)


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
        
        print(f"{species} max length = {SPECIES_MAX_LEN[species]}")


    for species in SPECIES:
        print(f"\n===== {species} =====")
        max_len = SPECIES_MAX_LEN[species]

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

            val_metrics, test_metrics, train_losses, val_losses = train_and_eval(X_train, y_train, X_test, y_test)
            import json

            run_result = {
                "species": species,
                "fold": fold,
                "K": K,
                "max_len": max_len,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "validation": val_metrics,
                "test": test_metrics,
                "train_loss": train_losses,
                "val_loss": val_losses
            }

            out_file = f"{RESULTS_DIR}/{species}_fold{fold}_K{K}_len{max_len}.json"

            with open(out_file, "w") as f:
                json.dump(run_result, f, indent=4)



            val_scores.append(val_metrics["acc"])
            test_scores.append(test_metrics["acc"])
            arch_txt = get_model_architecture(X_train.shape[1:])

            with open(f"{RESULTS_DIR}/{species}_K{K}_architecture.txt", "w") as f:
                f.write(arch_txt)
 

            print(f"Validation Acc: {val_metrics['acc']:.4f}")
            print(f"Test Acc:       {test_metrics['acc']:.4f}")
        print(f"\n{species} Mean Val Acc : {np.mean(val_scores):.4f}")
        print(f"{species} Mean Test Acc: {np.mean(test_scores):.4f}")
    
    import json

    with open(f"{RESULTS_DIR}/kmers_K{K}.json", "w") as f:
        json.dump({
            "K": K,
            "num_kmers": len(valid_kmers),
            "kmers": valid_kmers,
            "kmer_to_index": kmer_to_index
        }, f, indent=4)

