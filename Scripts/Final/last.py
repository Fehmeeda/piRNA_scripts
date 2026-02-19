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
DECISION_TYPE = "soft"   # or "hard"
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
PATIENCE=5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
import json
from datetime import datetime

RUN_NAME = f"{SPECIES}_{KMERTYPE}_{DECISION_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASE_SAVE_DIR = f"experiments/{RUN_NAME}"

print(f"Saving experiment to: {BASE_SAVE_DIR}") 
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
    "PATIENCE": PATIENCE
}
def compute_llr_stats(H_train):
    mean = H_train.mean(axis=0)   # per position mean
    std  = H_train.std(axis=0) + 1e-6
    return mean, std

def apply_llr_norm(H, mean, std):
    return (H - mean) / std

with open(os.path.join(BASE_SAVE_DIR, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=4)
# ============================================================
# LOAD PRECOMPUTED FEATURES
# ============================================================
def load_fold_features(species, fold, kmertype, decision_type):

    train_npz = np.load(f"encoded_features/{species}_{kmertype}_fold{fold}_train.npz")
    test_npz  = np.load(f"encoded_features/{species}_{kmertype}_fold{fold}_test.npz")

    X_train = train_npz["X_oh"]
    X_test  = test_npz["X_oh"]
    
    y_train = train_npz["y"]
    y_test  = test_npz["y"]
    Dtr = train_npz["X_d2v"]
    Dte = test_npz["X_d2v"]

    Dtr=Dtr.transpose(0,2,1) 
    Dte=Dte.transpose(0,2,1)

    # ---- Load HDV decision vectors ----
    hdv_train = np.load(f"hdv_vectors/{species}_fold{fold}_{kmertype}_train_hdv.npz")
    hdv_test  = np.load(f"hdv_vectors/{species}_fold{fold}_{kmertype}_test_hdv.npz")

    if decision_type == "soft":
        H_train = hdv_train["H_soft"]
        H_test  = hdv_test["H_soft"]
    else:
        H_train = hdv_train["H_hard"]
        H_test  = hdv_test["H_hard"]

    print("Loaded:")
    print("Weighted:", X_train.shape)
    print("DNA2Vec:", Dtr.shape)
    print("HDV:", H_train.shape)

    return X_train, Dtr, H_train, y_train, X_test, Dte, H_test, y_test


# ============================================================
# DATASET
# ============================================================
class FusionDataset(Dataset):
    #def __init__(self, X, D, H, y):
    def __init__(self, X,D,H, y):
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
    
    def __init__(self, input_shape,dna_shape, hdv_len, dropout=0.5):
        super().__init__()

        # Weighted CNN branch
        self.conv_w1 = nn.Conv2d(1, 32, kernel_size=(7,7), padding=3)
        self.bn_w1 = nn.BatchNorm2d(32)
        self.conv_w2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=2)
        self.bn_w2 = nn.BatchNorm2d(64)
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
            d = self.pool_d(F.relu(self.bn_d2(self.conv_d2(d))))
            self.flat_d = d.view(1, -1).shape[1]
        
        fusion_embed_dim = 128   # <-- choose shared dimension

        # Project CNN features → shared space
        self.proj_w = nn.Sequential(
            nn.Linear(self.flat_w, fusion_embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_embed_dim)
        )

        self.proj_d = nn.Sequential(
            nn.Linear(self.flat_d, fusion_embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_embed_dim)
        )
            
        # Project HDV → same space
        self.proj_h = nn.Sequential(
            nn.Linear(hdv_len, fusion_embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_embed_dim)
        )
            
        fusion_dim = 128 + 128 + 128
        self.fc1 = nn.Linear(fusion_dim, 264)
        self.fc2 = nn.Linear(264,64)
        self.fc3 = nn.Linear(64, 2)
        

    def forward(self, x_w, x_d, h):
        # Weighted branch
        w = self.pool_w(F.relu(self.bn_w1(self.conv_w1(x_w))))
        w = self.pool_w(F.relu(self.bn_w2(self.conv_w2(w))))
        w = torch.flatten(w, 1)
       
        w = self.proj_w(w)

        # DNA2Vec branch
        d = self.pool_d(F.relu(self.bn_d1(self.conv_d1(x_d))))
        d = self.pool_d(F.relu(self.bn_d2(self.conv_d2(d))))
        d = torch.flatten(d, 1)
        
        d = self.proj_d(d)
    
        
        h = self.proj_h(h)

        x = torch.cat([w,d,h], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)
# ============================================================
# TRAIN FUNCTION
# ============================================================    
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for  xb,db,hb, yb in loader:
        xb,db,hb, yb = xb.to(DEVICE), db.to(DEVICE), hb.to(DEVICE),  yb.to(DEVICE)
        out = model( xb,db,hb)

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
        for  xb,db,hb, yb in loader:
            xb,db,hb, yb = xb.to(DEVICE), db.to(DEVICE), hb.to(DEVICE),  yb.to(DEVICE)
            out = model(xb,db,hb)
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
        for xb,db,hb, yb in loader:
            xb,db,hb, yb = xb.to(DEVICE), db.to(DEVICE), hb.to(DEVICE),  yb.to(DEVICE)
            out = model(xb,db,hb)
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
 
    Xtr, Dtr, Htr, ytr, Xte,Dte, Hte, yte = load_fold_features(
    SPECIES, fold, KMERTYPE, DECISION_TYPE
    ) 
    llr_mean, llr_std = compute_llr_stats(Htr)

    Htr = apply_llr_norm(Htr, llr_mean, llr_std)
    Hte = apply_llr_norm(Hte, llr_mean, llr_std)

    train_ds = FusionDataset( Xtr, Dtr, Htr, ytr)
    test_ds  = FusionDataset( Xte,Dte, Hte, yte)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    hdv_dim = Htr.shape[1]
    input_shape = Xtr.shape[1:]
    dna_shape = Dtr.shape[1:]
    model = FusionNet(input_shape=input_shape, dna_shape=dna_shape, hdv_len=hdv_dim).to(DEVICE)

   
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    fold_dir = os.path.join(BASE_SAVE_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS): 
        train_loss = train_model(model, train_loader, optimizer, criterion)
        val_loss   = evaluate_loss(model, test_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    
    torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))
    with open(os.path.join(fold_dir, "model_architecture.txt"), "w") as f:
        f.write(str(model))
    
    history = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "epochs_trained": len(train_losses),
    
    }

    with open(os.path.join(fold_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    acc, auc, f1 = evaluate(model, test_loader)
    acc_train, auc_train, f1_train = evaluate(model, train_loader)

    results = {
        "accuracy": acc,
        "auc": auc,
        "f1_score": f1,
        "accuracy_train": acc_train,
        "auc_train": auc_train,
        "f1_score_train": f1_train
    }

    with open(os.path.join(fold_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Fold {fold} → Train ACC:{acc_train:.4f} AUC:{auc_train:.4f} F1:{f1_train:.4f}")
    print(f"Fold {fold} → ACC:{acc:.4f} AUC:{auc:.4f} F1:{f1:.4f}")

    all_results.append([acc, auc, f1, acc_train, auc_train, f1_train])

final_results = {
    "mean_accuracy": float(np.mean(all_results, axis=0)[0]),
    "mean_auc": float(np.mean(all_results, axis=0)[1]),
    "mean_f1": float(np.mean(all_results, axis=0)[2]),
    "mean_accuracy_train": float(np.mean(all_results, axis=0)[3]),
    "mean_auc_train": float(np.mean(all_results, axis=0)[4]),
    "mean_f1_train": float(np.mean(all_results, axis=0)[5]),
}

with open(os.path.join(BASE_SAVE_DIR, "cross_validation_results.json"), "w") as f:
    json.dump(final_results, f, indent=4)


print("\n===== FINAL RESULTS =====")
print(np.mean(all_results, axis=0))