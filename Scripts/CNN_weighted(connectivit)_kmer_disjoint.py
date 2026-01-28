"""# ============================================================
# CNN training for weighted disjoint k-mer one-hot encodings
# Works for Human / Mouse / Drosophila
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPECIES_FILES = {
    "Human": "Human/Human_kmer_weighted_disjoint_onehot.npz",
    "Mouse": "Mouse/Mouse_kmer_weighted_disjoint_onehot.npz",
    "Drosophila": "Drosophila/Drosophila_kmer_weighted_disjoint_onehot.npz"
}


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
# CNN MODEL (AUTO FC SIZE)
# ============================================================
class KmerCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.3)

        # --- auto compute FC size ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ============================================================
# TRAIN + EVAL
# ============================================================
def train_species(species, npz_file):

    print(f"\n==============================")
    print(f"Training CNN for {species}")
    print(f"==============================")

    data = np.load(npz_file)
    X, y = data["X"], data["y"]

    print("Input shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_ds = KmerDataset(X_train, y_train)
    test_ds = KmerDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = KmerCNN(input_shape=X.shape[1:]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ------------------ TRAIN ------------------
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(train_loader):.4f}")

    # ------------------ EVAL ------------------
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = torch.argmax(model(xb), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    print(f"{species} Test Accuracy: {acc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{species}_kmer_cnn.pt")
    print(f"Saved model: models/{species}_kmer_cnn.pt")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    for species, npz_path in SPECIES_FILES.items():
        train_species(species, npz_path)
"""

# ============================================================
# CNN with predefined folds + internal validation split
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
def get_disjoint_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq), k):
        chunk = seq[i:i+k]
        if len(chunk) < k:
            chunk += "N" * (k - len(chunk))
        kmers.append(chunk)
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
def encode_sequences(pos_file, neg_file, kmer_to_index):
    pos = read_fasta_txt(pos_file)
    neg = read_fasta_txt(neg_file)
    all_seqs = {**pos, **neg}

    padded, _ = pad_sequences(all_seqs)

    X, y = [], []

    for sid, seq in padded.items():
        kmers = get_disjoint_kmers(seq, K)
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

    for species in SPECIES:
        print(f"\n===== {species} =====")

        val_scores, test_scores = [], []

        for fold in FOLDS:
            print(f"\nFold {fold}")

            base = f"Splits/{species}/fold{fold}"

            X_train, y_train = encode_sequences(
                f"{base}/train_pos.txt", f"{base}/train_neg.txt", kmer_to_index
            )
            X_test, y_test = encode_sequences(
                f"{base}/test_pos.txt", f"{base}/test_neg.txt", kmer_to_index
            )

            val_acc, test_acc = train_and_eval(X_train, y_train, X_test, y_test)

            val_scores.append(val_acc)
            test_scores.append(test_acc)

            print(f"Validation Acc: {val_acc:.4f}")
            print(f"Test Acc:       {test_acc:.4f}")

        print(f"\n{species} Mean Val Acc : {np.mean(val_scores):.4f}")
        print(f"{species} Mean Test Acc: {np.mean(test_scores):.4f}")
