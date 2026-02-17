import torch
import numpy as np
from config import *
from model import FusionNet
from train_eval import train, evaluate
from torch.utils.data import DataLoader, TensorDataset, random_split

SPECIES = "Human"
FOLDS = range(5)
kmertype = 'overlap'

all_metrics = []
def load_encoded_fold(species, fold, kmertype=kmertype):
    train_file = f"encoded_features/{species}_{kmertype}_fold{fold}_train.npz"
    test_file  = f"encoded_features/{species}_{kmertype}_fold{fold}_test.npz"

    train_data = np.load(train_file, allow_pickle=True)
    test_data  = np.load(test_file, allow_pickle=True)

    Xtr_oh  = train_data["X_oh"]
    Xtr_d2v = train_data["X_d2v"]
    ytr     = train_data["y"]

    Xte_oh  = test_data["X_oh"]
    Xte_d2v = test_data["X_d2v"]
    yte     = test_data["y"]

    return Xtr_oh, Xtr_d2v, ytr, Xte_oh, Xte_d2v, yte

for fold in FOLDS:

    print(f"\n========== FOLD {fold} ==========")

    # ----------------------------
    # Load HDV Decision Vectors
    # ----------------------------
    #if USE_DECISION != None:
    train_npz = np.load(f"hdv_vectors/{SPECIES}_fold{fold}_{kmertype}_train_hdv.npz")
    test_npz  = np.load(f"hdv_vectors/{SPECIES}_fold{fold}_{kmertype}_test_hdv.npz")
    

    if USE_DECISION == "soft":
        X_train_dec = train_npz["H_soft"]
        X_test_dec  = test_npz["H_soft"]
    elif USE_DECISION == "hard":
        X_train_dec = train_npz["H_hard"]
        X_test_dec  = test_npz["H_hard"]
    '''else:
        X_train_dec = None
        X_test_dec  = None'''

    #y_train = train_npz["y"]
    #y_test  = test_npz["y"]
    

    # ----------------------------
    # Load OTHER embeddings (if enabled)
    # You already generate these earlier
    # ----------------------------
    X_train_oh, X_train_dna, y_train, X_test_oh, X_test_dna, y_test = load_encoded_fold(SPECIES, fold)
    '''if not USE_WEIGHTED:
        X_train_oh = None
        X_test_oh  = None

    if not USE_DNA2VEC:
        X_train_dna = None
        X_test_dna  = None'''
    
    

    # ----------------------------
    # Convert to tensors
    # ----------------------------
    
    def to_tensor(x):
        y=torch.tensor(x, dtype=torch.float32) if x is not None else None

        return y

    train_dataset = TensorDataset(
        to_tensor(X_train_oh)  if X_train_oh  is not None else torch.zeros(len(y_train),1),
        to_tensor(X_train_dna) if X_train_dna is not None else torch.zeros(len(y_train),1),
        to_tensor(X_train_dec) if X_train_dec is not None else torch.zeros(len(y_train),1),
        torch.tensor(y_train, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        to_tensor(X_test_oh)  if X_test_oh  is not None else torch.zeros(len(y_test),1),
        to_tensor(X_test_dna) if X_test_dna is not None else torch.zeros(len(y_test),1),
        to_tensor(X_test_dec) if X_test_dec is not None else torch.zeros(len(y_test),1),
        torch.tensor(y_test, dtype=torch.long)
    )

    # ---------------------------------
    # Split TRAIN into train + val
    # ---------------------------------
    val_ratio = 0.2
    train_size = int((1 - val_ratio) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # ----------------------------
    # Build NEW model per fold
    # ----------------------------
    print(X_train_oh.shape[1:])
    model = FusionNet(
        weighted_shape = X_train_oh.shape[1:]  if X_train_oh  is not None else None,
        dna2vec_shape  = X_train_dna.shape[1:] if X_train_dna is not None else None,
        decision_dim   = X_train_dec.shape[1]  if X_train_dec is not None else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT)
    criterion = torch.nn.CrossEntropyLoss()
    

    # ----------------------------
    # Train
    # ----------------------------
    save_dir = train(model, train_loader, val_loader, optimizer, criterion, "cpu")
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt"))
    metrics = evaluate(model, test_loader, save_dir)

    # ----------------------------
    # Evaluate Fold
    # ----------------------------
    
    all_metrics.append(metrics)