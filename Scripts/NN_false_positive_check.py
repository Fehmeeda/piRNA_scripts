'''import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ============================
# CONFIG (DO NOT CHANGE MODEL)
# ============================
K = 3
ALPHABET = ["A", "C", "G", "T"]
THRESHOLD = 0.5

OUTDIR = "Human_FP_Position_Specific_Kmer"
os.makedirs(OUTDIR, exist_ok=True)

# ============================
# FILES (YOUR FILES)
# ============================
X_FILE = "Human_X_kmer_prob.npy"
Y_FILE = "Human_y.npy"

POS_SEQ_FILE = "Datasets/Human_posi_samples.txt"
NEG_SEQ_FILE = "Datasets/Human_nega_samples.txt"

# ============================
# FASTA / TXT READER
# ============================
def read_fasta_txt(path):
    sequences = []
    with open(path) as f:
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.upper()
        if seq:
            sequences.append(seq)
    return np.array(sequences)

# ============================
# LOAD SEQUENCES
# ============================
pos_seqs = read_fasta_txt(POS_SEQ_FILE)
neg_seqs = read_fasta_txt(NEG_SEQ_FILE)

print("Positive sequences:", len(pos_seqs))
print("Negative sequences:", len(neg_seqs))

# ============================
# LOAD FEATURES + LABELS
# ============================
X = np.load(X_FILE)
y = np.load(Y_FILE)

sequences = np.concatenate([pos_seqs, neg_seqs])

assert len(X) == len(y) == len(sequences), "❌ X, y, sequence length mismatch"

# ============================
# TRAIN / TEST SPLIT (KEEP ALIGNMENT)
# ============================
X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
    X, y, sequences,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# STANDARDIZE (AS YOU ALREADY DO)
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# YOUR MODEL (UNCHANGED)
# ============================
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ============================
# TRAIN
# ============================
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ============================
# PREDICT
# ============================
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= THRESHOLD).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ============================
# EXTRACT FALSE POSITIVES
# ============================
fp_mask = (y_test == 0) & (y_pred == 1)
fp_sequences = seq_test[fp_mask]

print("False Positives:", len(fp_sequences))

# Save FP sequences
with open(os.path.join(OUTDIR, "false_positive_sequences.txt"), "w") as f:
    for i, seq in enumerate(fp_sequences):
        f.write(f">FP_{i}\n{seq}\n")

if len(fp_sequences) == 0:
    print("⚠️ No false positives found. Exiting.")
    exit()

# ============================
# POSITION-SPECIFIC K-MER HISTOGRAM
# ============================
KMERS = ["".join(p) for p in itertools.product(ALPHABET, repeat=K)]
KMER_INDEX = {k: i for i, k in enumerate(KMERS)}

L = len(fp_sequences[0])
NUM_POS = L - K + 1

pos_counts = np.zeros((NUM_POS, len(KMERS)))

for seq in fp_sequences:
    for pos in range(NUM_POS):
        kmer = seq[pos:pos+K]
        if kmer in KMER_INDEX:
            pos_counts[pos, KMER_INDEX[kmer]] += 1

# Normalize per position
pos_freq = pos_counts / pos_counts.sum(axis=1, keepdims=True)

# Save matrix
np.save(os.path.join(OUTDIR, "FP_position_specific_kmer.npy"), pos_freq)

# ============================
# HEATMAP
# ============================
plt.figure(figsize=(16, 6))
plt.imshow(pos_freq.T, aspect="auto", cmap="viridis")
plt.colorbar(label="Frequency")
plt.xlabel("Position")
plt.ylabel("k-mer")
plt.yticks(range(len(KMERS)), KMERS)
plt.title("False Positive Position-Specific 3-mer Histogram")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "FP_kmer_position_heatmap.png"))
plt.close()

print("✅ FP position-specific k-mer analysis completed")
print("📁 Results saved in:", OUTDIR)
'''
'''
import os
import itertools
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =============================
# CONFIG
# =============================
K = 3
ALPHABET = ["A", "C", "G", "T"]   # use T since your data uses T
THRESHOLD = 0.5

OUT_DIR = "FP_position_kmer_trim_to_min"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# FILES
# =============================
X_FILE = "Human_X_kmer_prob.npy"
Y_FILE = "Human_y.npy"

POS_SEQ_FILE = "Datasets/Human_posi_samples.txt"
NEG_SEQ_FILE = "Datasets/Human_nega_samples.txt"
# =============================
# FILES PER SPECIES
# =============================
SPECIES_FILES = {
    "Human": {
        "X": "Human_X_kmer_prob.npy",
        "y": "Human_y.npy",
        "pos": "Datasets/Human_posi_samples.txt",
        "neg": "Datasets/Human_nega_samples.txt",
    },
    "Mouse": {
        "X": "Mouse_X_kmer_prob.npy",
        "y": "Mouse_y.npy",
        "pos": "Datasets/Mouse_posi_samples.txt",
        "neg": "Datasets/Mouse_nega_samples.txt",
    },
    "Drosophila": {
        "X": "Drosophila_X_kmer_prob.npy",
        "y": "Drosophila_y.npy",
        "pos": "Datasets/Drosophila_posi_samples.txt",
        "neg": "Datasets/Drosophila_nega_samples.txt",
    }
}


# =============================
# FASTA / TXT READER
# =============================
def read_fasta_or_txt(filepath):
    sequences = []
    seq = ""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq.upper())
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq.upper())
    return np.array(sequences)

# =============================
# K-MER SETUP
# =============================
KMERS = ["".join(p) for p in itertools.product(ALPHABET, repeat=K)]
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# POSITION-SPECIFIC KMER (OVERLAP)
# =============================
def kmer_freq_overlap_trim(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)

    for seq in sequences:
        seq = seq[:target_len]   # 🔑 TRIM TO MIN LENGTH
        for i in range(positions):
            kmer = seq[i:i+K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat

# =============================
# POSITION-SPECIFIC KMER (DISJOINT)
# =============================
def kmer_freq_disjoint_trim(sequences, target_len):
    positions = target_len // K
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)

    for seq in sequences:
        seq = seq[:target_len]   # 🔑 TRIM TO MIN LENGTH
        for i in range(positions):
            kmer = seq[i*K:(i+1)*K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1

    return mat

# =============================
# STACKED HISTOGRAM (PLOTLY)
# =============================
def plot_histogram(matrix, title, outfile):
    fig = go.Figure()

    for i, kmer in enumerate(KMERS):
        fig.add_bar(
            x=list(range(matrix.shape[0])),
            y=matrix[:, i],
            name=kmer
        )

    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Frequency",
        barmode="stack",
        height=700,
        legend_title="3-mers"
    )

    fig.write_html(outfile)
    print(f"✅ Saved: {outfile}")

# =============================
# MAIN
# =============================
'''
'''def main():

    # ---- Load sequences ----
    pos_seqs = read_fasta_or_txt(POS_SEQ_FILE)
    neg_seqs = read_fasta_or_txt(NEG_SEQ_FILE)

    sequences = np.concatenate([pos_seqs, neg_seqs])

    # ---- Load X and y ----
    X = np.load(X_FILE)
    y = np.load(Y_FILE)

    assert len(X) == len(y) == len(sequences), "Length mismatch!"

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
        X, y, sequences,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---- Standardize ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---- Model ----
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # ---- Predict ----
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # =============================
    # FALSE POSITIVES
    # =============================
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_sequences = seq_test[fp_mask]

    print("\nFalse Positives:", len(fp_sequences))

    if len(fp_sequences) == 0:
        print("No false positives found.")
        return

    # ---- Find MIN length ----
    min_len = min(len(seq) for seq in fp_sequences)
    print("Minimum FP sequence length:", min_len)

    # =============================
    # HISTOGRAMS (TRIM TO MIN)
    # =============================
    fp_overlap = kmer_freq_overlap_trim(fp_sequences, min_len)
    fp_disjoint = kmer_freq_disjoint_trim(fp_sequences, min_len)
    # ===== SAVE FREQUENCY TABLES =====
    fp_ov_df = save_kmer_position_table(
        fp_overlap,
        KMERS,
        f"{OUT_DIR}/FP_overlap_trim_{min_len}_table.csv"
    )

    fp_dis_df = save_kmer_position_table(
        fp_disjoint,
        KMERS,
        f"{OUT_DIR}/FP_disjoint_trim_{min_len}_table.csv"
    )


    plot_histogram(
        fp_overlap,
        f"False Positive Overlapping {K}-mer (Trimmed to {min_len})",
        f"{OUT_DIR}/FP_overlap_trim_{min_len}.html"
    )

    plot_histogram(
        fp_disjoint,
        f"False Positive Disjoint {K}-mer (Trimmed to {min_len})",
        f"{OUT_DIR}/FP_disjoint_trim_{min_len}.html"
    )

    print("\n✔ FP position-specific k-mer analysis completed")

def main():

    for species, files in SPECIES_FILES.items():

        print(f"\n🧬 ===== Processing {species} =====")

        species_out = os.path.join(OUT_DIR, species)
        os.makedirs(species_out, exist_ok=True)

        # ---- Load sequences ----
        pos_seqs = read_fasta_or_txt(files["pos"])
        neg_seqs = read_fasta_or_txt(files["neg"])
        sequences = np.concatenate([pos_seqs, neg_seqs])

        # ---- Load X and y ----
        X = np.load(files["X"])
        y = np.load(files["y"])

        assert len(X) == len(y) == len(sequences), \
            f"Length mismatch for {species}"

        # ---- Train / test split ----
        X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
            X, y, sequences,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ---- Standardize ----
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---- Model ----
        model = Sequential([
            Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        # ---- Predict ----
        y_prob = model.predict(X_test).ravel()
        y_pred = (y_prob >= THRESHOLD).astype(int)

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))

        # =============================
        # FALSE POSITIVES
        # =============================
        fp_mask = (y_test == 0) & (y_pred == 1)
        fp_sequences = seq_test[fp_mask]

        print(f"False Positives in {species}: {len(fp_sequences)}")

        if len(fp_sequences) == 0:
            print("No false positives — skipping k-mer analysis")
            continue

        # ---- Minimum length ----
        min_len = min(len(seq) for seq in fp_sequences)
        print("Min FP length:", min_len)

        # ---- K-mer matrices ----
        fp_overlap = kmer_freq_overlap_trim(fp_sequences, min_len)
        fp_disjoint = kmer_freq_disjoint_trim(fp_sequences, min_len)

        # ---- Save tables ----
        save_kmer_position_table(
            fp_overlap,
            KMERS,
            f"{species_out}/FP_overlap_trim_{min_len}_table.csv"
        )

        save_kmer_position_table(
            fp_disjoint,
            KMERS,
            f"{species_out}/FP_disjoint_trim_{min_len}_table.csv"
        )

        # ---- Plots ----
        plot_histogram(
            fp_overlap,
            f"{species} FP Overlapping {K}-mer (L={min_len})",
            f"{species_out}/FP_overlap_trim_{min_len}.html"
        )

        plot_histogram(
            fp_disjoint,
            f"{species} FP Disjoint {K}-mer (L={min_len})",
            f"{species_out}/FP_disjoint_trim_{min_len}.html"
        )

        print(f"✔ Finished {species}")

    print("\n🎯 All species processed successfully")

def save_kmer_position_table(matrix, kmers, outfile):
    """
    Convert (positions x kmers) matrix
    → (kmers x positions) table and save as CSV.
    Rows = k-mers
    Columns = positions
    Values = raw frequencies
    """

    table = matrix.T   # transpose → rows = kmers

    df = pd.DataFrame(
        table,
        index=kmers,
        columns=[f"pos_{i}" for i in range(table.shape[1])]
    )

    df.to_csv(outfile)
    print(f"✅ Saved frequency table: {outfile}")
    return df

# =============================
if __name__ == "__main__":
    main()'''

import os
import itertools
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =============================
# CONFIG
# =============================
K = 3
ALPHABET = ["A", "C", "G", "T"]
THRESHOLD = 0.5

OUT_DIR = "NN_FP_FN_POSITION_KMER"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# FILES PER SPECIES
# =============================
SPECIES_FILES = {
    "Human": {
        "X": "Human_X_kmer_prob.npy",
        "y": "Human_y.npy",
        "pos": "Datasets/Human_posi_samples.txt",
        "neg": "Datasets/Human_nega_samples.txt",
    },
    "Mouse": {
        "X": "Mouse_X_kmer_prob.npy",
        "y": "Mouse_y.npy",
        "pos": "Datasets/Mouse_posi_samples.txt",
        "neg": "Datasets/Mouse_nega_samples.txt",
    },
    "Drosophila": {
        "X": "Drosophila_X_kmer_prob.npy",
        "y": "Drosophila_y.npy",
        "pos": "Datasets/Drosophila_posi_samples.txt",
        "neg": "Datasets/Drosophila_nega_samples.txt",
    }
}

# =============================
# FASTA / TXT READER
# =============================
def read_fasta_or_txt(filepath):
    sequences = []
    seq = ""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq.upper())
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq.upper())
    return np.array(sequences)

# =============================
# KMER SETUP
# =============================
KMERS = ["".join(p) for p in itertools.product(ALPHABET, repeat=K)]
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# =============================
# POSITION-SPECIFIC KMER
# =============================
def kmer_freq_overlap_trim(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)

    for seq in sequences:
        seq = seq[:target_len]
        for i in range(positions):
            kmer = seq[i:i+K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1
    return mat

def kmer_freq_disjoint_trim(sequences, target_len):
    positions = target_len // K
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)

    for seq in sequences:
        seq = seq[:target_len]
        for i in range(positions):
            kmer = seq[i*K:(i+1)*K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1
    return mat

# =============================
# SAVE KMER TABLE
# =============================
def save_kmer_position_table(matrix, kmers, outfile):
    table = matrix.T
    df = pd.DataFrame(
        table,
        index=kmers,
        columns=[f"pos_{i}" for i in range(table.shape[1])]
    )
    df.to_csv(outfile)
    print(f"✅ Saved: {outfile}")

# =============================
# STACKED HISTOGRAM
# =============================
def plot_histogram(matrix, title, outfile):
    fig = go.Figure()
    for i, kmer in enumerate(KMERS):
        fig.add_bar(
            x=list(range(matrix.shape[0])),
            y=matrix[:, i],
            name=kmer
        )
    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Frequency",
        barmode="stack",
        height=700
    )
    fig.write_html(outfile)

# =============================
# MAIN
# =============================
def main():

    for species, files in SPECIES_FILES.items():

        print(f"\n🧬 ===== Processing {species} =====")

        outdir = os.path.join(OUT_DIR, species)
        os.makedirs(outdir, exist_ok=True)

        # ---- Load data ----
        X = np.load(files["X"])
        y = np.load(files["y"])

        pos_seqs = read_fasta_or_txt(files["pos"])
        neg_seqs = read_fasta_or_txt(files["neg"])
        sequences = np.concatenate([pos_seqs, neg_seqs])

        # ---- Split ----
        X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
            X, y, sequences,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ---- Scale ----
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---- Model ----
        model = Sequential([
            Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        # ---- Predict ----
        y_prob = model.predict(X_test).ravel()
        y_pred = (y_prob >= THRESHOLD).astype(int)

        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))

        # =============================
        # CONFUSION MATRIX (LABELED)
        # =============================
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,5))
        plt.imshow(cm, cmap="Blues")

        plt.xticks([0,1], ["Predicted non-piRNA", "Predicted piRNA"])
        plt.yticks([0,1], ["True non-piRNA", "True piRNA"])

        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{labels[i][j]}\n{cm[i,j]}",
                         ha="center", va="center", fontsize=12)

        plt.title(f"{species} NN Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=300)
        plt.close()

        # =============================
        # FP & FN ANALYSIS
        # =============================
        for label, mask in {
            "FP": (y_test == 0) & (y_pred == 1),
            "FN": (y_test == 1) & (y_pred == 0)
        }.items():

            seqs = seq_test[mask]
            print(f"{label} count: {len(seqs)}")

            if len(seqs) == 0:
                continue

            min_len = min(len(s) for s in seqs)

            overlap = kmer_freq_overlap_trim(seqs, min_len)
            disjoint = kmer_freq_disjoint_trim(seqs, min_len)

            save_kmer_position_table(
                overlap, KMERS,
                f"{outdir}/{label}_overlap_trim_{min_len}.csv"
            )
            save_kmer_position_table(
                disjoint, KMERS,
                f"{outdir}/{label}_disjoint_trim_{min_len}.csv"
            )

            plot_histogram(
                overlap,
                f"{species} {label} Overlap {K}-mer",
                f"{outdir}/{label}_overlap.html"
            )
            plot_histogram(
                disjoint,
                f"{species} {label} Disjoint {K}-mer",
                f"{outdir}/{label}_disjoint.html"
            )

        print(f"✔ Finished {species}")

    print("\n🎯 ALL SPECIES COMPLETED")

# =============================
if __name__ == "__main__":
    main()
