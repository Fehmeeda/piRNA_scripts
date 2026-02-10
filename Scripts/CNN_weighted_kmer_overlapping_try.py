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
EPOCHS = 10
LR = 1e-3
DROPOUT=0.5
kmertype="overlap"
DEVICE = "cpu"
K = 3
DECISION_TYPE='soft_llr'
#hard or soft_llr
FOLDS = range(5)
PATIENCE=5
MIN_DELTA=1e-5
WEIGHT_DECAY=1e-3
def load_species_probs(species):
  
  
   base = f"output_position_kmer_probabilities"
   pos = pd.read_csv(f"{base}/{species}_pos_{kmertype}_prob.csv", index_col=0)
   neg = pd.read_csv(f"{base}/{species}_neg_{kmertype}_prob.csv", index_col=0)
  
   return pos, neg


SPECIES = ["Human"]


# ============================================================
# KMER FUNCTIONS
# ============================================================
def get_overlapping_kmers(seq, k=3):
   kmers = []
   for i in range(0, len(seq) - k + 1):
       kmers.append(seq[i:i+k])
   return kmers


def get_disjoint_kmers(seq, k=3):
   kmers = []
   for i in range(0, len(seq) - k + 1, k):  # step = k
       kmers.append(seq[i:i+k])
   return kmers


#If I just remove the N, it will not include any of the Kmers with N in the combination (all kmers product)
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


#Include N in alphabet
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


#✅ Do NOT include N in alphabet
#✅ Do NOT include N in kmer dictionary
#✅ Skip kmers containing N
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


def hard_decision_vector(seq, pos_prob, neg_prob, k, kmertype):
   L = len(seq)


   if kmertype == "overlap":
       step = 1
   elif kmertype == "disjoint":
       step = k
   else:
       raise ValueError("kmertype must be 'overlap' or 'disjoint'")
  
   positions = (L - k) // step + 1


   vec = np.zeros(positions, dtype=np.float32)
   #print(pos_prob)


   idx = 0
   for i in range(0, L - k + 1, step):


       kmer = seq[i:i+k]


       if kmer not in pos_prob.index:
           idx += 1
           continue


       if pos_prob.loc[kmer, f"pos_{idx}"] > neg_prob.loc[kmer, f"pos_{idx}"]:
           vec[idx] = 1.0


       idx += 1
       #print(f'pos0 and first kmer in posi:{pos_prob.loc['AAA','pos_0']}')


   '''s = vec.sum()
   if s > 0:
       vec /= s'''
   return vec
def soft_decision_vector_llr(seq, pos_prob, neg_prob, k, kmertype):
   L = len(seq)
   step = 1 if kmertype == "overlap" else k
   positions = (L - k) // step + 1


   vec = np.zeros(positions, dtype=np.float32)
   eps = 1e-6
   idx=0


   for i in range(0, L - k + 1, step):


       kmer = seq[i:i+k]


       if kmer not in pos_prob.index:
           idx += 1
           continue


       p_pos = pos_prob.loc[kmer, f"pos_{idx}"]
  
       p_neg = neg_prob.loc[kmer, f"pos_{idx}"]


       vec[idx] = np.log((p_pos + eps) / (p_neg + eps))
       idx += 1


   return vec


def encode_sequences(pos_file, neg_file, kmer_to_index, cnn_len, hdv_len, pos_prob, neg_prob):
   pos = read_fasta_txt(pos_file)
   neg = read_fasta_txt(neg_file)
   all_seqs = {**pos, **neg}


   #print(f"Loaded {len(pos)} positive and {len(neg)} negative sequences.")


   X, y , ids, H = [], [], [], []


   for sid, seq in all_seqs.items():


   # ===== CNN sequence (PAD to max_len) =====
       seq_cnn = seq
       if len(seq_cnn) < cnn_len:
           seq_cnn = seq_cnn + "N" * (cnn_len - len(seq_cnn))
          
       else:
           seq_cnn = seq_cnn[:cnn_len]


       if kmertype=="overlap":
           kmers = get_overlapping_kmers(seq_cnn, K)
       else:
           kmers= get_disjoint_kmers(seq_cnn,K)
       #print(kmers)
       X.append(weighted_one_hot_kmers(kmers, kmer_to_index))


       # ===== HDV sequence (TRUNCATE to min_len) =====
       '''seq_hdv = seq[:hdv_len]
       h = soft_decision_vector_llr(seq_hdv, pos_prob, neg_prob, K)
       H.append(h)'''
       seq_hdv = seq[:hdv_len]


       if DECISION_TYPE == "soft_llr":
           h = soft_decision_vector_llr(seq_hdv, pos_prob, neg_prob, K,kmertype)
       elif DECISION_TYPE == "hard":
           h = hard_decision_vector(seq_hdv, pos_prob, neg_prob, K,kmertype)
       else:
           raise ValueError("Unknown decision type")


       H.append(h)




       y.append(1 if sid in pos else 0)
       ids.append(sid)


   return np.stack(X), np.array(y), np.array(ids), np.stack(H)


# ============================================================
# DATASET
# ============================================================
class KmerDataset(Dataset):
   def __init__(self, X, H, y):
       self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
       #H = np.array(H, dtype=np.float32)
       #H = np.tanh(H)
  
       self.H = torch.tensor(H, dtype=torch.float32)
       self.y = torch.tensor(y, dtype=torch.long)


   def __len__(self):
       return len(self.y)


   def __getitem__(self, idx):
       return self.X[idx], self.H[idx], self.y[idx]






def get_model_architecture(input_shape, hdv_len):
   model = KmerCNN(input_shape, hdv_len)
   return str(model)
'''def get_model_architecture(input_shape, hdv_len):
   num_kmers = input_shape[0]
   seq_len = input_shape[1]
   model = KmerCNN(num_kmers, seq_len, hdv_len)
   return str(model)'''




# ============================================================
# CNN MODEL
# ============================================================
class KmerCNN(nn.Module):
   def __init__(self, input_shape, hdv_len):
       super().__init__()


       self.conv1 = nn.Conv2d(1, 32, 7, stride=1,padding=2)
       self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
       #self.bn1 = nn.GroupNorm(8, 32)
       #self.bn2 = nn.GroupNorm(8, 64)
       #self.bn1 = nn.BatchNorm2d(64)
       #self.bn2 = nn.BatchNorm2d(128)
       #self.bn1 = nn.BatchNorm1d(128)
       #self.bn2 = nn.BatchNorm1d(32)


       self.pool = nn.MaxPool2d((1, 2))  # only pool along sequence axis
       self.dropout = nn.Dropout(DROPOUT)


       #self.conv_h = nn.Sequential(
   #nn.Conv1d(1, 32, kernel_size=3, padding=1),
   #nn.GroupNorm(8, 32),
   #nn.ReLU(),
   #nn.AdaptiveMaxPool1d(1)
       #)


       with torch.no_grad():
           dummy = torch.zeros(1,1,*input_shape)
           #x = F.relu(self.bn1(self.conv1(dummy)))
           #x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
           #x = self.pool(F.relu(self.bn2(self.conv2(x))))
           x = self.pool(F.relu(self.conv1(dummy)))
           x = self.pool(F.relu(self.conv2(x)))
           #x = F.relu(self.bn2(self.conv2(x)))
           self.flat = x.view(1,-1).shape[1]


       # ===== DENSE LAYERS =====
       self.fc1 = nn.Linear(self.flat + hdv_len, 48)
       self.fc2 = nn.Linear(48, 16)
       #self.fc_h = nn.Sequential(nn.Linear(hdv_len, 128), nn.BatchNorm1d(128),nn.ReLU())
      
    # NEW LAYER
       self.fc3 = nn.Linear(16, 2)        # FINAL OUTPUT
       #self.fc_h = nn.Linear(hdv_len, 96)


   def forward(self, x,h):




       #x = self.pool(F.relu(self.bn1(self.conv1(x))))
       #x = self.pool(F.relu(self.bn2(self.conv2(x))))
       #x = F.relu(self.bn1(self.conv1(x)))
       #x = F.relu(self.bn2(self.conv2(x)))
       x = self.pool(F.relu(self.conv1(x)))
       x = self.pool(F.relu(self.conv2(x)))


       x = torch.flatten(x, 1)
       x = F.normalize(x, p=2, dim=1)
       h = F.normalize(h, p=2, dim=1)
       #h = self.fc_h(h)


       #gate = self.h_gate(h)
       #h = h * gate
       #h = self.h_proj(h)
       #h = h.unsqueeze(1)          # [B, 1, hdv_len]
       #h = self.conv_h(h)
       #h = h.squeeze(-1)          # [B, 32]
      




       x = torch.cat([x, h], dim=1)
      
       x = self.dropout(F.relu(self.fc1(x)))
       x = self.dropout(F.relu(self.fc2(x)))   # NEW
       #x = self.dropout(F.relu(self.bn1(self.fc1(x))))
       #x = self.dropout(F.relu(self.bn2(self.fc2(x))))


       return self.fc3(x)

'''
class KmerCNN(nn.Module):
   def __init__(self, num_kmers, seq_len, hdv_len):
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
       self.dropout = nn.Dropout(DROPOUT)
       # infer size automatically
       with torch.no_grad():
           dummy = torch.zeros(1, num_kmers, seq_len)
           dummy = self.pool(self.relu(self.conv1(dummy)))
           dummy = self.relu(self.conv2(dummy))
           flat_dim = dummy.numel()


       self.fc1 = nn.Linear(flat_dim + hdv_len, 48)
       self.fc2 = nn.Linear(48, 16)
       self.fc3 = nn.Linear(16, 2)


   def forward(self, x,h):
       # x: B × num_kmers × positions
       x = self.pool(self.relu(self.conv1(x)))
       x = self.relu(self.conv2(x))
       x = x.flatten(1)


       # fusion
       x = torch.cat([x, h], dim=1)


       x = self.dropout(self.relu(self.fc1(x)))
       x = self.dropout(self.relu(self.fc2(x)))
       return self.fc3(x)
'''

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
def train_and_eval(train_X, train_H, train_y, train_ids,
                  test_X, test_H, test_y):




   X_tr, X_val, H_tr, H_val, y_tr, y_val, ids_tr, ids_val = train_test_split(
       train_X,
       train_H,
       train_y,
       train_ids,
       test_size=0.2,
       stratify=train_y,
       random_state=42
   )
   #X_tr, H_tr, y_tr = train_X, train_H, train_y






   train_loader = DataLoader(KmerDataset(X_tr, H_tr, y_tr), BATCH_SIZE, shuffle=True)
   val_loader   = DataLoader(KmerDataset(X_val, H_val, y_val), BATCH_SIZE)
   test_loader  = DataLoader(KmerDataset(test_X, test_H, test_y), BATCH_SIZE)


  
   print("Class distribution:")
   print("Train:", np.bincount(y_tr))
   print("Val:  ", np.bincount(y_val))


   print(f"Train samples: {len(y_tr)}, Val samples: {len(y_val)}, Test samples: {len(test_y)}")


   num_kmers = train_X.shape[1]
   seq_len   = train_X.shape[2]
   print(f"Number of kmers: {num_kmers}, Sequence length: {seq_len}")


  
   hdv_len = train_H.shape[1]
   print(hdv_len)
   #model = KmerCNN(num_kmers, seq_len, hdv_len).to(DEVICE)
   model = KmerCNN(train_X.shape[1:], hdv_len).to(DEVICE)


   opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
   #opt = torch.optim.Adam(model.parameters(), lr=LR)
   crit = nn.CrossEntropyLoss()
   #crit = nn.CrossEntropyLoss(label_smoothing=0.03)


   train_losses, val_losses = [], []
   train_losses=[]


   best_val = float("inf")
   patience = PATIENCE
   min_delta = MIN_DELTA
   counter = 0


   for epoch in range(EPOCHS):


       # ===== TRAIN =====
       model.train()
       epoch_loss = 0


       for xb, hb, yb in train_loader:
           xb, hb, yb = xb.to(DEVICE), hb.to(DEVICE), yb.to(DEVICE)
           loss = crit(model(xb, hb), yb)
           opt.zero_grad()


           loss.backward()
           opt.step()
           epoch_loss += loss.item()


       train_losses.append(epoch_loss / len(train_loader))


       # ===== VALIDATION =====
       model.eval()
       val_loss = 0
       with torch.no_grad():
           for xb, hb, yb in val_loader:
               xb, hb, yb = xb.to(DEVICE), hb.to(DEVICE), yb.to(DEVICE)
               val_loss += crit(model(xb, hb), yb).item()




       val_loss /= len(val_loader)
       val_losses.append(val_loss)


       # ===== EARLY STOPPING =====
       if val_loss < best_val - min_delta:
           best_val = val_loss
           counter = 0
           torch.save(model.state_dict(), f"{species}_model.pt")
       else:
           counter += 1


       if counter >= patience:
           print(f"Early stopping at epoch {epoch}")
           break
       '''np.savetxt(f"{RESULTS_DIR}/{species}_fold{fold}_train_ids.txt",
       ids_tr,
       fmt="%s")


       np.savetxt(
           f"{RESULTS_DIR}/{species}_fold{fold}_val_ids.txt",
           ids_val,
           fmt="%s"
       )'''
   #torch.save(model.state_dict(), f"{species}_model.pt")




   model.load_state_dict(torch.load(f"{species}_model.pt"))




   # ===== METRIC EVALUATION =====
   def evaluate(loader):
       model.eval()
       probs, preds, true = [], [], []


       with torch.no_grad():
           for xb, hb, yb in loader:
               xb, hb = xb.to(DEVICE), hb.to(DEVICE)
               out = model(xb, hb)
               best_t = np.linspace(0.3, 0.7, 21)


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


           X_train, y_train, train_ids, H_train = encode_sequences(
               f"{base}/train_pos.txt",
               f"{base}/train_neg.txt",
               kmer_to_index,
               cnn_len,
               hdv_len,
               POS_PROB,
               NEG_PROB
           )
           print(H_train)


           X_test, y_test, test_ids, H_test = encode_sequences(
               f"{base}/test_pos.txt",
               f"{base}/test_neg.txt",
               kmer_to_index,
               cnn_len,
               hdv_len,
               POS_PROB,
               NEG_PROB
           )


           val_metrics, test_metrics, train_losses, val_losses = train_and_eval(X_train, H_train, y_train, train_ids, X_test,  H_test,  y_test)
           #test_metrics, train_losses = train_and_eval(X_train, H_train, y_train, train_ids, X_test,  H_test,  y_test)


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
               "kmertype": kmertype,
               "decision type":DECISION_TYPE,
               "Specie":SPECIES,
               "patience": PATIENCE,
               "min_delta":MIN_DELTA,
               'weight_decay':WEIGHT_DECAY,
           }


           out_file = f"{RESULTS_DIR}/{species}_fold{fold}_K{K}_len{max_len}.json"


           with open(out_file, "w") as f:
               json.dump(run_result, f, indent=4)






           val_scores.append(val_metrics["acc"])
           test_scores.append(test_metrics["acc"])
           arch_txt = get_model_architecture(X_train.shape[1:],H_train.shape[1])


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
           "kmer_to_index": kmer_to_index,
           'mean_test_accuracy':f"{np.mean(test_scores):.4f}",
       }, f, indent=4)




