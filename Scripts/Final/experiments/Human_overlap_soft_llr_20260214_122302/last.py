import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import shutil
import sys


# ============================================================
# CONFIG
# ============================================================
SPECIES = "Human"
FOLDS = range(5)
KMERTYPE = "overlap"
DECISION_TYPE = "soft_llr"   # or "hard"
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
import json
from datetime import datetime

RUN_NAME = f"{SPECIES}_{KMERTYPE}_{DECISION_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASE_SAVE_DIR = f"experiments/{RUN_NAME}"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
# Save a copy of this script into experiment folder
script_path = os.path.abspath(sys.argv[0])
shutil.copy(script_path, os.path.join(BASE_SAVE_DIR, "last.py"))

config_dict = {
    "SPECIES": SPECIES,
    "KMERTYPE": KMERTYPE,
    "DECISION_TYPE": DECISION_TYPE,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "DEVICE": DEVICE,
    "PATIENCE": 5
}

with open(os.path.join(BASE_SAVE_DIR, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=4)
# ============================================================
# LOAD PRECOMPUTED FEATURES
# ============================================================
def load_fold_features(species, fold, kmertype, decision_type):

    train_npz = np.load(f"/Users/fehmeeda/Desktop/PiRNA/encoded_features/{species}_{kmertype}_fold{fold}_train.npz")
    test_npz  = np.load(f"/Users/fehmeeda/Desktop/PiRNA/encoded_features/{species}_{kmertype}_fold{fold}_test.npz")

    X_train = train_npz["X_oh"]
    X_test  = test_npz["X_oh"]
    

    y_train = train_npz["y"]
    y_test  = test_npz["y"]
    Dtr = train_npz["X_d2v"]
    Dte = test_npz["X_d2v"]

    X_train=X_train.transpose(0,2,1) 
    X_test=X_test.transpose(0,2,1)
    

    # ---- Load HDV decision vectors ----
    hdv_train = np.load(f"/Users/fehmeeda/Desktop/PiRNA/hdv_vectors/{species}_fold{fold}_{kmertype}_train_hdv.npz")
    hdv_test  = np.load(f"/Users/fehmeeda/Desktop/PiRNA/hdv_vectors/{species}_fold{fold}_{kmertype}_test_hdv.npz")

    if decision_type == "soft_llr":
        H_train = hdv_train["H_soft"]
        H_test  = hdv_test["H_soft"]
    else:
        H_train = hdv_train["H_hard"]
        H_test  = hdv_test["H_hard"]

    print("Loaded:")
    print("Weighted:", X_train.shape)
    print("HDV:", H_train.shape)

    return X_train, Dtr, H_train, y_train, X_test, Dte, H_test, y_test

def create_train_val_split(X, D, H, y, val_ratio=0.2):
    Xtr, Xval, Dtr, Dval, Htr, Hval, ytr, yval = train_test_split(
        X, D, H, y,
        test_size=val_ratio,
        stratify=y,
        random_state=42
    )
    return Xtr, Dtr, Htr, ytr, Xval, Dval, Hval, yval

# ============================================================
# DATASET
# ============================================================
class FusionDataset(Dataset):
    def __init__(self, X, D, H, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # weighted
        self.D = torch.tensor(D, dtype=torch.float32).unsqueeze(1)  # dna2vec
        self.H = torch.tensor(H, dtype=torch.float32)               # hdv
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.D[idx], self.H[idx], self.y[idx]


# ============================================================
# MODEL
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
class FusionNet(nn.Module):
    def __init__(self, input_shape, dna_shape, hdv_len, dropout=0.5):
        super().__init__()

        # Weighted CNN branch
        self.conv_w1 = nn.Conv2d(1, 32, kernel_size=(7,7), padding=2)
        self.bn_w1 = nn.BatchNorm2d(32)
        self.bn_w2 = nn.BatchNorm2d(64)
        self.conv_w2 = nn.Conv2d(32, 64, kernel_size=(5,5), padding=1)
        self.pool_w  = nn.MaxPool2d((2))
        
        # DNA2Vec CNN branch
        self.conv_d1 = nn.Conv2d(1, 32, kernel_size=(5,5), padding=2)
        self.bn_d1 = nn.BatchNorm2d(32)
        self.conv_d2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.pool_d  = nn.MaxPool2d((2))
        
        self.dropout = nn.Dropout(dropout)

        # compute flattened dims
        with torch.no_grad():
            dummy_w = torch.zeros(1, 1, *input_shape)
            w = self.pool_w(F.relu(self.bn_w1(self.conv_w1(dummy_w))))
            w = self.pool_w(F.relu(self.bn_w2(self.conv_w2(w))))
            self.flat_w = w.view(1, -1).shape[1]

            dummy_d = torch.zeros(1, 1, *dna_shape)
            d = self.pool_d(F.relu(self.bn_d1(self.conv_d1(dummy_d))))
            d = self.pool_d(F.relu((self.bn_d2(self.conv_d2(d)))))
            self.flat_d = d.view(1, -1).shape[1]

        # Fully connected layers after fusion
        fusion_dim = self.flat_w + self.flat_d + hdv_len
        self.fc1 = nn.Linear(fusion_dim, 48)
        self.fc2 = nn.Linear(48, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x_w, x_d, h):
        # Weighted branch
        w = self.pool_w(F.relu(self.bn_w1(self.conv_w1(x_w))))
        w = self.pool_w(F.relu(self.bn_w2(self.conv_w2(w))))
        w = torch.flatten(w, 1)
        w = F.normalize(w, p=2, dim=1)

        # DNA2Vec branch
        d = self.pool_d(F.relu(self.bn_d1(self.conv_d1(x_d))))
        d = self.pool_d(F.relu(self.bn_d2(self.conv_d2(d))))
        d = torch.flatten(d, 1)
        d = F.normalize(d, p=2, dim=1)

        # Decision vector
        h = F.normalize(h, p=2, dim=1)

        # Concatenate all features
        x = torch.cat([w, d, h], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)
# ============================================================
# TRAIN FUNCTION
# ============================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for xb, db, hb, yb in loader:
        xb, db, hb, yb = xb.to(DEVICE), db.to(DEVICE), hb.to(DEVICE), yb.to(DEVICE)
        out = model(xb, db, hb)

        optimizer.zero_grad()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for xb, db, hb, yb in loader:
            xb, db, hb, yb = xb.to(DEVICE), db.to(DEVICE), hb.to(DEVICE), yb.to(DEVICE)
            out = model(xb, db, hb)
            loss = criterion(out, yb)
            total_loss += loss.item()

    return total_loss / len(loader)
# ============================================================
# EVALUATE
# ============================================================
def evaluate(model, loader):
    model.eval()
    preds, probs, true = [], [], []

    with torch.no_grad():
        for xb, db, hb, yb in loader:
            xb, db, hb, yb = xb.to(DEVICE), db.to(DEVICE), hb.to(DEVICE), yb.to(DEVICE)
            out = model(xb, db, hb)
            p = torch.softmax(out, dim=1)[:, 1]
            

            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            true.extend(yb.numpy())

    acc = accuracy_score(true, preds)
    auc = roc_auc_score(true, probs)
    f1  = f1_score(true, preds)

    return acc, auc, f1


# ============================================================
# MAIN CROSS-VALIDATION
# ============================================================
all_results = []

for fold in FOLDS:
    print(f"\n===== Fold {fold} =====")

    Xtr, Dtr, Htr, ytr, Xte, Dte, Hte, yte = load_fold_features(
    SPECIES, fold, KMERTYPE, DECISION_TYPE
    )   

    # -------- create validation split --------
    Xtr, Dtr, Htr, ytr, Xval, Dval, Hval, yval = create_train_val_split(Xtr, Dtr, Htr, ytr)

    train_ds = FusionDataset(Xtr, Dtr, Htr, ytr)
    val_ds   = FusionDataset(Xval, Dval, Hval, yval)
    test_ds  = FusionDataset(Xte, Dte, Hte, yte)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    hdv_dim = Htr.shape[1]
    input_shape = Xtr.shape[1:]
    dna_shape = Dtr.shape[1:]
    model = FusionNet(input_shape=input_shape, dna_shape=dna_shape, hdv_len=hdv_dim).to(DEVICE)

    weight_decay=1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    early_stop = EarlyStopping(patience=5)
    fold_dir = os.path.join(BASE_SAVE_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):  # allow ES to decide when to stop
        train_loss = train_model(model, train_loader, optimizer, criterion)
        val_loss   = evaluate_loss(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if early_stop.step(val_loss, model):
            print("Early stopping triggered ✔")
            break

    # restore best model
    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)
    else:
        print("Warning: No improvement detected, using last epoch weights.")

    torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))
    with open(os.path.join(fold_dir, "model_architecture.txt"), "w") as f:
        f.write(str(model))
    
    history = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "epochs_trained": len(train_losses),
    "weight_decay":weight_decay
    }

    with open(os.path.join(fold_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    acc, auc, f1 = evaluate(model, test_loader)

    results = {
        "accuracy": acc,
        "auc": auc,
        "f1_score": f1
    }

    with open(os.path.join(fold_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Fold {fold} → ACC:{acc:.4f} AUC:{auc:.4f} F1:{f1:.4f}")
    all_results.append([acc, auc, f1])

final_results = {
    "mean_accuracy": float(np.mean(all_results, axis=0)[0]),
    "mean_auc": float(np.mean(all_results, axis=0)[1]),
    "mean_f1": float(np.mean(all_results, axis=0)[2])
}

with open(os.path.join(BASE_SAVE_DIR, "cross_validation_results.json"), "w") as f:
    json.dump(final_results, f, indent=4)
print("\n===== FINAL RESULTS =====")
print(np.mean(all_results, axis=0))