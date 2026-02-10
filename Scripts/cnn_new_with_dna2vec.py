# ============================================================
# CNN with predefined folds + internal validation split
# ============================================================

import pandas as pd
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
from pirna import read_fasta_txt
import os

# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DROPOUT=0.3

# ===============================
# Load prebuilt dna2vec embeddings
# ===============================
dna2vec_data = np.load("all_3mer_embeddings_with_null.npz")
DNA2VEC_EMB = dna2vec_data["embeddings"]   # (65, 100)
DNA2VEC_DIM = DNA2VEC_EMB.shape[1]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
K = 3
FOLDS = range(5)
def load_species_probs(species):
    
    base = f"output_position_kmer_probabilities"
    pos = pd.read_csv(f"{base}/{species}_pos_overlap_prob.csv", index_col=0)
    neg = pd.read_csv(f"{base}/{species}_neg_overlap_prob.csv", index_col=0)
    return pos, neg

SPECIES = ["Human","Mouse","Drosophila"]

# ============================================================
# KMER FUNCTIONS
# ============================================================
def get_overlapping_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq) - k + 1):
        kmers.append(seq[i:i+k])
    return kmers

#If I just remove the N, it will not include any of the Kmers with N in the combination (all kmers product)
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
def dna2vec_sequence_matrix(seq, k, kmer_to_index):
    kmers = get_overlapping_kmers(seq, k)
    vecs = []

    for kmer in kmers:
        idx = kmer_to_index.get(kmer, 0)
        vecs.append(DNA2VEC_EMB[idx])

    return np.array(vecs, dtype=np.float32)   # (L−k+1, embed_dim)


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

def hard_decision_vector(seq, pos_prob, neg_prob, k):
    L = len(seq)
    
    positions = L - k + 1
    
    vec = np.zeros(positions, dtype=np.float32)
    #print(pos_prob)

    for i in range(positions):
        kmer = seq[i:i+k]
        
        if kmer not in pos_prob.index:
            
            continue

        if pos_prob.loc[kmer, f"pos_{i}"] > neg_prob.loc[kmer, f"pos_{i}"]:
            vec[i] = 1.0

        #print(f'pos0 and first kmer in posi:{pos_prob.loc['AAA','pos_0']}')

    
    return vec
def soft_decision_vector_llr(seq, pos_prob, neg_prob, k):
    L = len(seq)
    positions = L - k + 1

    vec = np.zeros(positions, dtype=np.float32)
    eps = 1e-6

    for i in range(positions):
        #print(positions)
        kmer = seq[i:i+k]

        if kmer not in pos_prob.index:
            vec[i] = 0.0
            continue
        #print(f"pos_{i},kmer:{kmer},{pos_prob.loc[kmer,f'pos_{i}']}")
    
        p_pos = pos_prob.loc[kmer, f"pos_{i}"]
        p_neg = neg_prob.loc[kmer, f"pos_{i}"]
        #print(f'pos0 and first kmer in posi:{pos_prob.loc['AAA','pos_0']}')
        #print(f'pos_{i}, kmer: {kmer},pos_prob{p_pos}, neg_prob{p_neg}')

        # LLR
        vec[i] = np.log((p_pos + eps) / (p_neg + eps))

    return vec

def encode_sequences(pos_file, neg_file, kmer_to_index, cnn_len):

    pos = read_fasta_txt(pos_file)
    neg = read_fasta_txt(neg_file)
    all_seqs = {**pos, **neg}

    X_oh, X_d2v, y, ids = [], [], [], []

    for sid, seq in all_seqs.items():

        seq = seq[:cnn_len].ljust(cnn_len, "N")
        kmers = get_overlapping_kmers(seq, K)

        # weighted one-hot (kmer × position)
        X_oh.append(weighted_one_hot_kmers(kmers, kmer_to_index))

        # dna2vec matrix (position × embed_dim)
        X_d2v.append(dna2vec_sequence_matrix(seq, K, kmer_to_index))

        y.append(1 if sid in pos else 0)
        ids.append(sid)

    return (
        np.stack(X_oh),
        np.stack(X_d2v),
        np.array(y),
        np.array(ids)
    )

# ============================================================
# DATASET
# ============================================================
class KmerDataset(Dataset):
    def __init__(self, X_oh, X_d2v, y):
        self.X_oh = torch.tensor(X_oh, dtype=torch.float32).unsqueeze(1)
        self.X_d2v = torch.tensor(X_d2v, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        #self.mask = (self.X_d2v.abs().sum(dim=-1) > 0).float()
# shape: (B, 1, L)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_oh[idx], self.X_d2v[idx], self.y[idx]
        #eturn self.X_oh[idx], self.X_d2v[idx], self.mask[idx], self.y[idx]

def get_model_architecture(oh_shape, d2v_shape):
    model = KmerCNN(oh_shape, d2v_shape)
    return str(model)


# ============================================================
# CNN MODEL
# ============================================================
class KmerCNN(nn.Module):
    def __init__(self, oh_shape, d2v_shape):
        super().__init__()

        # --- One-hot CNN ---
        self.conv_oh1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv_oh2 = nn.Conv2d(32, 64, 3, padding=1)

        # --- dna2vec CNN ---
        self.conv_d1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv_d2 = nn.Conv2d(32, 64, 3, padding=1)

        self.dropout = nn.Dropout(DROPOUT)
        self.pool = nn.MaxPool2d(2)

        with torch.no_grad():
            oh = torch.zeros(1, 1, *oh_shape)
            d2 = torch.zeros(1, 1, *d2v_shape)

            oh = F.relu(self.conv_oh1(oh))
            oh = self.pool(oh)
            oh = F.relu(self.conv_oh2(oh))
            oh = self.pool(oh)

            d2 = F.relu(self.conv_d1(d2))
            d2 = self.pool(d2)
            d2 = F.relu(self.conv_d2(d2))
            d2 = self.pool(d2)

            self.flat_oh = oh.view(1, -1).shape[1]
            self.flat_d2 = d2.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flat_oh + self.flat_d2, 64)
        self.fc2 = nn.Linear(64, 2)
        

    def forward(self, x_oh, x_d2v):

        x_oh = F.relu(self.conv_oh1(x_oh))
        x_oh = self.pool(x_oh)
        x_oh = F.relu(self.conv_oh2(x_oh))
        x_oh = self.pool(x_oh)
        x_oh = x_oh.flatten(1)
        x_oh = F.normalize(x_oh, p=2, dim=1)

         # ---- dna2vec branch ----
        '''mask = mask.unsqueeze(-1)      # (B,1,L,1)
        x_d2v = x_d2v * mask'''

        x_d2v = F.relu(self.conv_d1(x_d2v))
        x_d2v = self.pool(x_d2v)
        x_d2v = F.relu(self.conv_d2(x_d2v))
        x_d2v = self.pool(x_d2v)
        
        x_d2v = x_d2v.flatten(1)
        x_d2v = F.normalize(x_d2v, p=2, dim=1)

        x = torch.cat([x_oh, x_d2v], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
    
        return self.fc2(x)


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_and_eval(train_X, train_D, train_y, train_ids,
                   test_X, test_D, test_y):



    X_tr, X_val, D_tr, D_val, y_tr, y_val, ids_tr, ids_val = train_test_split(
    train_X,
    train_D,
    train_y,
    train_ids,
    test_size=0.2,
    stratify=train_y,
    random_state=42
)



    train_loader = DataLoader(
    KmerDataset(X_tr, D_tr, y_tr),
    BATCH_SIZE,
    shuffle=True
    )

    val_loader = DataLoader(
        KmerDataset(X_val, D_val, y_val),
        BATCH_SIZE
    )

    test_loader = DataLoader(
        KmerDataset(test_X, test_D, test_y),
        BATCH_SIZE
    )


    
    print("Class distribution:")
    print("Train:", np.bincount(y_tr))
    print("Val:  ", np.bincount(y_val))

    print(f"Train samples: {len(y_tr)}, Val samples: {len(y_val)}, Test samples: {len(test_y)}")

    #num_kmers = train_X.shape[1]
    #seq_len   = train_X.shape[2]
    #print(f"Number of kmers: {num_kmers}, Sequence length: {seq_len}")

    #model = KmerCNN(seq_len, num_kmers).to(DEVICE)
    
    model = KmerCNN(
        oh_shape=train_X.shape[1:],
        d2v_shape=train_D.shape[1:]
    ).to(DEVICE)

    opt = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)
    crit = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    best_val = float("inf")
    patience = 5
    counter = 0

    for epoch in range(EPOCHS):

        # ===== TRAIN =====
        model.train()
        epoch_loss = 0

        for xb, db, yb in train_loader:
            xb = xb.to(DEVICE)
            db = db.to(DEVICE)
           
            yb = yb.to(DEVICE)

            out = model(xb, db)
            loss = crit(out, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, db, yb in val_loader:
                xb = xb.to(DEVICE)
                db = db.to(DEVICE)
                
                yb = yb.to(DEVICE)

                val_loss += crit(model(xb, db), yb).item()



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
        np.savetxt(f"{RESULTS_DIR}/{species}_fold{fold}_train_ids.txt",
        ids_tr,
        fmt="%s")

        np.savetxt(
            f"{RESULTS_DIR}/{species}_fold{fold}_val_ids.txt",
            ids_val,
            fmt="%s"
        )


    model.load_state_dict(torch.load(f"{species}_model.pt"))


    # ===== METRIC EVALUATION =====
    def evaluate(loader):
        model.eval()
        probs, preds, true = [], [], []

        with torch.no_grad():
            for xb, db, yb in loader:
                xb = xb.to(DEVICE)
                db = db.to(DEVICE)
                

                out = model(xb, db)
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

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
RESULTS_DIR = "results_cnn_dna_2_vec"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# MAIN CV LOOP
# ============================================================
if __name__ == "__main__":

    valid_kmers = generate_valid_kmers(K)
    kmer_to_index = {k:i for i,k in enumerate(valid_kmers)}
    
    SPECIES_MAX_LEN = {}
    SPECIES_MIN_LEN = {}
    

    for species in SPECIES:
        POS_PROB, NEG_PROB = load_species_probs(species)

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
        SPECIES_MIN_LEN[species] = min(lengths)
        
        print(f"{species} max length = {SPECIES_MAX_LEN[species]}")
        print(f"{species} min length = {SPECIES_MIN_LEN[species]}")

        cnn_len = SPECIES_MAX_LEN[species]
        hdv_len = SPECIES_MIN_LEN[species]


    for species in SPECIES:
        print(f"\n===== {species} =====")
        POS_PROB, NEG_PROB = load_species_probs(species)
        max_len = SPECIES_MAX_LEN[species]

        val_scores, test_scores = [], []

        for fold in FOLDS:
            print(f"\nFold {fold}")

            base = f"Splits/{species}/fold{fold}"
            print(species)

            X_train, D_train, y_train, train_ids = encode_sequences(
            f"{base}/train_pos.txt",
            f"{base}/train_neg.txt",
            kmer_to_index,
            cnn_len
        )

            

            X_test, D_test, y_test, test_ids = encode_sequences(
            f"{base}/test_pos.txt",
            f"{base}/test_neg.txt",
            kmer_to_index,
            cnn_len
        )


            val_metrics, test_metrics, train_losses, val_losses = train_and_eval(
    X_train, D_train, y_train, train_ids,
    X_test, D_test, y_test
)
 

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
                "val_loss": val_losses,
                "dropout": DROPOUT,
            }

            out_file = f"{RESULTS_DIR}/{species}_fold{fold}_K{K}_len{max_len}.json"

            with open(out_file, "w") as f:
                json.dump(run_result, f, indent=4)



            val_scores.append(val_metrics["acc"])
            test_scores.append(test_metrics["acc"])
            arch_txt = get_model_architecture(
    X_train.shape[1:],
    D_train.shape[1:]
)

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

