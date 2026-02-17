from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config import *
import torch
import csv
import json
import os
import shutil
from datetime import datetime

import inspect
import config

config_path = inspect.getfile(config)

min_delta = 1e-3

def train(model, train_loader, val_loader, optimizer, criterion, device):

    # ===== Create experiment folder =====
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"runs_new/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    patience_counter = 0

    history = []   # store losses

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0
        total_samples = 0

        for w, d, dec, y in train_loader:

            w = w.to(device)
            d = d.to(device)
            dec = dec.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(w, d, dec)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        # save losses in memory
        history.append([epoch, train_loss, val_loss])

        # ===== EARLY STOPPING =====
        if val_loss < best_val - min_delta:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # ===== Save loss history to CSV =====
    with open(f"{save_dir}/loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(history)
    
    shutil.copy(config_path, f"{save_dir}/config.py")


    return save_dir

def evaluate(model, loader, save_dir):

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    device="cpu"

    with torch.no_grad():
        for w, d, dec, y in loader:
            w = w.to(device)
            d = d.to(device)
            dec = dec.to(device)

            out = model(w, d, dec)
            prob = torch.softmax(out,1)[:,1]

            y_true.extend(y.numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred),
        "recall": recall_score(y_true,y_pred),
        "f1": f1_score(y_true,y_pred),
        "roc_auc": roc_auc_score(y_true,y_prob)
    }

    print(metrics)

    # Save metrics
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

def evaluate_loss(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for w, d, dec, y in loader:

            w = w.to(device)
            d = d.to(device)
            dec = dec.to(device)
            y = y.to(device)

            out = model(w, d, dec)
            loss = criterion(out, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples



