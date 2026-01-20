import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ============================
# CONFIG
# ============================
K = 3
ALPHABET = ["A", "C", "G", "T"]
THRESHOLD = 0.5

BASE_OUTDIR = "SVM_FP_KMER_RESULTS"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# ============================
# FILES PER SPECIES
# ============================
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

# ============================
# READ FASTA / TXT
# ============================
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

# ============================
# KMER SETUP
# ============================
KMERS = ["".join(p) for p in itertools.product(ALPHABET, repeat=K)]
KMER_TO_IDX = {k: i for i, k in enumerate(KMERS)}

# ============================
# FP OVERLAPPING KMER MATRIX
# ============================
def kmer_freq_overlap_trim(sequences, target_len):
    positions = target_len - K + 1
    mat = np.zeros((positions, len(KMERS)), dtype=np.int32)

    for seq in sequences:
        seq = seq[:target_len]   # trim longer sequences
        for i in range(positions):
            kmer = seq[i:i+K]
            if set(kmer) <= set(ALPHABET):
                mat[i, KMER_TO_IDX[kmer]] += 1
    return mat

# ============================
# FP DISJOINT KMER MATRIX
# ============================
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

# ============================
# SAVE KMER TABLE (CSV)
# ============================
def save_kmer_position_table(matrix, kmers, outfile):
    table = matrix.T   # rows = kmers, cols = positions

    df = pd.DataFrame(
        table,
        index=kmers,
        columns=[f"pos_{i}" for i in range(table.shape[1])]
    )

    df.to_csv(outfile)
    print(f"Saved table: {outfile}")
    return df
import plotly.graph_objects as go

def plot_histogram_plotly(matrix, title, outfile):
    fig = go.Figure()

    for i, kmer in enumerate(KMERS):
        fig.add_bar(
            x=list(range(matrix.shape[0])),
            y=matrix[:, i],
            name=kmer,
            hovertemplate=(
                f"K-mer: {kmer}<br>"
                "Position: %{x}<br>"
                "Frequency: %{y}<extra></extra>"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Frequency",
        barmode="stack",
        height=700,
        legend_title=f"{K}-mers"
    )

    fig.write_html(outfile)
    print(f"Saved interactive plot: {outfile}")

# ============================
# PLOT STACKED BAR
# ============================
def plot_histogram(matrix, title, outfile):
    plt.figure(figsize=(10,6))
    bottom = np.zeros(matrix.shape[0])

    for i, kmer in enumerate(KMERS):
        plt.bar(range(matrix.shape[0]), matrix[:, i], bottom=bottom, label=kmer)
        bottom += matrix[:, i]

    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# ============================
# MAIN LOOP
# ============================
for species, files in SPECIES_FILES.items():

    print(f"\n====== Processing {species} ======")

    outdir = os.path.join(BASE_OUTDIR, species)
    os.makedirs(outdir, exist_ok=True)

    # ---- Load data ----
    X = np.load(files["X"])
    y = np.load(files["y"])

    pos_seqs = read_fasta_or_txt(files["pos"])
    neg_seqs = read_fasta_or_txt(files["neg"])
    sequences = np.concatenate([pos_seqs, neg_seqs])

    assert len(X) == len(y) == len(sequences), "Length mismatch!"

    # ---- Train-test split ----
    X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
        X, y, sequences,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---- SVM model ----
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    svm_model.fit(X_train, y_train)

    # ---- Predictions ----
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:,1]

    # ---- Metrics ----
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(report)

    with open(os.path.join(outdir, "svm_report.txt"), "w") as f:
        f.write(report)

    # ---- Confusion matrix ----

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{species} SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.close()

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.close()

    # ============================
    # FALSE POSITIVE ANALYSIS
    # ============================
    fp_mask = (y_test == 0) & (y_pred == 1)
    fp_sequences = seq_test[fp_mask]

    print(f"False Positives in {species}: {len(fp_sequences)}")

    if len(fp_sequences) == 0:
        print("No false positives → skipping k-mer analysis")
        continue

    # ---- minimum length ----
    min_len = min(len(seq) for seq in fp_sequences)
    print("Minimum FP length:", min_len)

    # ---- compute matrices ----
    fp_overlap = kmer_freq_overlap_trim(fp_sequences, min_len)
    fp_disjoint = kmer_freq_disjoint_trim(fp_sequences, min_len)

    # ---- save tables ----
    save_kmer_position_table(
        fp_overlap, KMERS,
        f"{outdir}/FP_overlap_trim_{min_len}.csv"
    )

    save_kmer_position_table(
        fp_disjoint, KMERS,
        f"{outdir}/FP_disjoint_trim_{min_len}.csv"
    )

    # ---- plots ----
    plot_histogram(
        fp_overlap,
        f"{species} FP Overlap K-mer",
        f"{outdir}/FP_overlap_plot.png"
    )

    plot_histogram(
        fp_disjoint,
        f"{species} FP Disjoint K-mer",
        f"{outdir}/FP_disjoint_plot.png"
    )
    plot_histogram_plotly(
        fp_overlap,
        f"{species} False Positive Overlapping {K}-mer",
        f"{outdir}/FP_overlap_plot.html"
    )

    plot_histogram_plotly(
        fp_disjoint,
        f"{species} False Positive Disjoint {K}-mer",
        f"{outdir}/FP_disjoint_plot.html"
    )


    print(f"Finished FP k-mer analysis for {species}")

print("\n✔ All species finished")
