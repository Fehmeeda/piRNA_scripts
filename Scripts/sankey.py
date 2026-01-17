
'''import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================
# Read FASTA-like txt
# ============================
def read_fasta_txt(filename):
    sequences = []
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                seq = ''
            else:
                seq += line.upper()
        if seq:
            sequences.append(seq)
    return sequences


# ============================
# k-mer frequency (normalized)
# ============================
def kmer_freq(seqs, k=3):
    counter = Counter()
    total = 0
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N' not in kmer:
                counter[kmer] += 1
                total += 1
    return {k: v / total for k, v in counter.items()}


def top_k(freq_dict, k=10):
    return sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:k]


# ============================
# FILES
# ============================
species = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}


# ============================
# STEP 1: k-mer difference
# ============================
for sp, (posf, negf) in species.items():
    print(f"\n================ {sp} =================")

    pos_seqs = read_fasta_txt(posf)
    neg_seqs = read_fasta_txt(negf)

    freq_pos = kmer_freq(pos_seqs)
    freq_neg = kmer_freq(neg_seqs)

    print("\nTop-10 Positive k-mers:")
    for k, v in top_k(freq_pos):
        print(f"{k}: {v:.4f}")

    print("\nTop-10 Negative k-mers:")
    for k, v in top_k(freq_neg):
        print(f"{k}: {v:.4f}")


# ============================
# STEP 2: HUMAN t-SNE SPLIT VISUALIZATION
# ============================
tsne_data = np.load("Human/Humantsne_features_kmer_weighted_onehot.npz")
X_tsne = tsne_data["X"]
y = tsne_data["y"]

pos_idx = np.where(y == 1)[0]

plt.figure(figsize=(7, 5))
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], s=5, label="Positive")
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], s=5, label="Negative")

# provisional threshold (WE WILL CHANGE THIS)
threshold = np.median(X_tsne[pos_idx, 0])

plt.axvline(threshold, color="black", linestyle="--", label=f"x = {threshold:.1f}")

plt.legend()
plt.title("Human t-SNE with provisional split")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()

print(f"\nCurrent Human split at x = {threshold:.2f}")
'''
'''
import numpy as np
from collections import Counter
import plotly.graph_objects as go

# ============================
# Read FASTA-like txt
# ============================
def read_fasta_txt(filename):
    sequences = []
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                seq = ''
            else:
                seq += line.upper()
        if seq:
            sequences.append(seq)
    return sequences


# ============================
# k-mer frequency (normalized)
# ============================
def kmer_freq(seqs, k=3):
    counter = Counter()
    total = 0
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N' not in kmer:
                counter[kmer] += 1
                total += 1
    return {k: v / total for k, v in counter.items()}


# ============================
# k-mer enrichment (DIRECTION!)
# ============================
def kmer_enrichment(freq_A, freq_B, eps=1e-6):
    """
    log2( A / B )
    > 0 : enriched in A
    < 0 : enriched in B
    """
    all_kmers = set(freq_A) | set(freq_B)
    enrich = {}

    for k in all_kmers:
        a = freq_A.get(k, eps)
        b = freq_B.get(k, eps)
        enrich[k] = np.log2((a + eps) / (b + eps))

    return enrich


# ============================
# Sankey plot (FINAL, MEANINGFUL)
# ============================
def sankey_enrichment(freq_A, freq_B, label_A, label_B, title, top_k=10):
    enrich = kmer_enrichment(freq_A, freq_B)

    # strongest differences
    top = sorted(enrich.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    left_nodes = []
    right_nodes = []
    links_src = []
    links_tgt = []
    links_val = []

    for k, val in top:
        if val > 0:
            left_nodes.append(f"{k}\n{label_A} ↑")
        else:
            right_nodes.append(f"{k}\n{label_B} ↑")

    labels = left_nodes + right_nodes

    # connect all left to all right proportionally
    for i, k1 in enumerate(left_nodes):
        for j, k2 in enumerate(right_nodes):
            links_src.append(i)
            links_tgt.append(len(left_nodes) + j)
            links_val.append(1)

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=15),
        link=dict(source=links_src, target=links_tgt, value=links_val)
    ))

    fig.update_layout(title_text=title, font_size=11)

    out = title.replace(" ", "_") + ".html"
    fig.write_html(out)
    print(f"Sankey saved → {out}")


# ============================
# FILES
# ============================
species = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}


# ============================
# MAIN
# ============================
for sp, (posf, negf) in species.items():
    print(f"\nProcessing {sp}")

    pos_seqs = read_fasta_txt(posf)
    neg_seqs = read_fasta_txt(negf)

    freq_pos = kmer_freq(pos_seqs)
    freq_neg = kmer_freq(neg_seqs)

    # ---- POSITIVE vs NEGATIVE (ALL SPECIES) ----
    sankey_enrichment(
        freq_pos,
        freq_neg,
        "Positive",
        "Negative",
        title=f"{sp} Positive vs Negative"
    )

    # ---- HUMAN SPECIAL CASE ----
    if sp == "Human":
        tsne = np.load("Human/Humantsne_features_kmer_weighted_onehot.npz")
        X_tsne = tsne["X"]
        y = tsne["y"]

        pos_idx = np.where(y == 1)[0]

        # CONFIRMED THRESHOLD
        threshold = 60

        left_idx = pos_idx[X_tsne[pos_idx, 0] <= threshold]
        right_idx = pos_idx[X_tsne[pos_idx, 0] > threshold]

        left_seqs = [pos_seqs[i] for i in left_idx]
        right_seqs = [pos_seqs[i] for i in right_idx]

        freq_left = kmer_freq(left_seqs)
        freq_right = kmer_freq(right_seqs)

        sankey_enrichment(
            freq_left,
            freq_right,
            "Left Positive",
            "Right Positive",
            title="Human Left vs Right Positive"
        )

        sankey_enrichment(
            freq_right,
            freq_neg,
            "Right Positive",
            "Negative",
            title="Human Right Positive vs Negative"
        )
'''
'''
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================
# Read FASTA-like txt
# ============================
def read_fasta_txt(filename):
    sequences = []
    with open(filename, 'r') as f:
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                seq = ''
            else:
                seq += line.upper()
        if seq:
            sequences.append(seq)
    return sequences


# ============================
# k-mer frequency (normalized)
# ============================
def kmer_freq(seqs, k=3):
    counter = Counter()
    total = 0
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if 'N' not in kmer:
                counter[kmer] += 1
                total += 1
    return {k: v / total for k, v in counter.items()}


# ============================
# Enrichment computation
# ============================
def kmer_enrichment(freq_pos, freq_neg, eps=1e-9):
    kmers = set(freq_pos) | set(freq_neg)
    rows = []
    for k in kmers:
        fp = freq_pos.get(k, eps)
        fn = freq_neg.get(k, eps)
        log2fc = np.log2((fp + eps) / (fn + eps))
        rows.append((k, fp, fn, log2fc))

    df = pd.DataFrame(
        rows,
        columns=["kmer", "freq_positive", "freq_negative", "log2_enrichment"]
    )
    return df.sort_values("log2_enrichment", ascending=False)


# ============================
# Diverging bar plot
# ============================
def plot_diverging(df, species, top_k=10):
    top_pos = df.head(top_k)
    top_neg = df.tail(top_k)

    plot_df = pd.concat([top_neg, top_pos])
    plot_df = plot_df.sort_values("log2_enrichment")

    plt.figure(figsize=(8, 6))
    colors = ["#d62728" if v < 0 else "#2ca02c"
              for v in plot_df["log2_enrichment"]]

    plt.barh(
        plot_df["kmer"],
        plot_df["log2_enrichment"],
        color=colors
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("log₂(Positive / Negative)")
    plt.title(f"{species}: k-mer enrichment")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out = f"plots/{species}_kmer_enrichment.png"
    plt.savefig(out, dpi=300)
    plt.show()

    print(f"Saved plot → {out}")


# ============================
# FILES
# ============================
species_files = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}


# ============================
# MAIN
# ============================
for sp, (posf, negf) in species_files.items():
    print(f"\n================ {sp} =================")

    pos_seqs = read_fasta_txt(posf)
    neg_seqs = read_fasta_txt(negf)

    freq_pos = kmer_freq(pos_seqs)
    freq_neg = kmer_freq(neg_seqs)

    df = kmer_enrichment(freq_pos, freq_neg)

    # ---- PRINT TABLE ----
    print("\nTop-10 Positive-enriched k-mers:")
    print(df.head(10)[["kmer", "log2_enrichment"]].to_string(index=False))

    print("\nTop-10 Negative-enriched k-mers:")
    print(df.tail(10)[["kmer", "log2_enrichment"]].to_string(index=False))

    # ---- PLOT ----
    plot_diverging(df, sp)
'''
'''
import os
from itertools import product
from collections import Counter
import plotly.graph_objects as go

# =============================
# CONFIG
# =============================
K = 10          # Top-K k-mers
KMER_SIZE = 3   # 3-mers

SPECIES = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}

# =============================
# FASTA READER
# =============================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, 'r') as f:
        seq_id, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = seq.upper()
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences[seq_id] = seq.upper()
    return sequences


# =============================
# K-MER EXTRACTION
# =============================
def get_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# =============================
# K-MER FREQUENCY
# =============================
def kmer_frequency(seqs, k=3):
    counter = Counter()
    for seq in seqs.values():
        counter.update(get_kmers(seq, k))
    return counter


# =============================
# NORMALIZATION
# =============================
def normalize(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


# =============================
# RANKING
# =============================
def rank_kmers(freq_dict):
    sorted_items = sorted(freq_dict.items(),
                           key=lambda x: x[1],
                           reverse=True)
    return {kmer: rank + 1 for rank, (kmer, _) in enumerate(sorted_items)}


# =============================
# SANKEY CREATION
# =============================
def make_rank_change_sankey(species_name, pos_file, neg_file):

    print(f"\nProcessing {species_name}...")

    # Read sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # Count & normalize
    pos_freq = normalize(kmer_frequency(pos_seqs, KMER_SIZE))
    neg_freq = normalize(kmer_frequency(neg_seqs, KMER_SIZE))

    # Rank
    pos_rank = rank_kmers(pos_freq)
    neg_rank = rank_kmers(neg_freq)

    # Top-K Positive k-mers
    topK_pos = sorted(pos_freq.items(),
                      key=lambda x: x[1],
                      reverse=True)[:K]

    topK_kmers = [k for k, _ in topK_pos]

    # Nodes
    left_nodes = [f"P_Rank_{i}" for i in range(1, K + 1)]
    right_nodes = [f"N_Rank_{i}" for i in range(1, K + 1)]
    nodes = left_nodes + right_nodes + ["N_NotTop10"]

    node_index = {n: i for i, n in enumerate(nodes)}

    # Links
    sources, targets, values, labels = [], [], [], []

    for kmer in topK_kmers:
        p_rank = pos_rank[kmer]

        if kmer in neg_rank and neg_rank[kmer] <= K:
            target = f"N_Rank_{neg_rank[kmer]}"
        else:
            target = "N_NotTop10"

        sources.append(node_index[f"P_Rank_{p_rank}"])
        targets.append(node_index[target])
        values.append(pos_freq[kmer])   # use 1 for rank-only Sankey
        labels.append(kmer)

    # Sankey Plot
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels
        )
    ))

    fig.update_layout(
        title=f"{species_name}: Rank-Change Sankey (Top-{K} Positive 3-mers)",
        font_size=11
    )

    # Save
    os.makedirs(species_name, exist_ok=True)
    out_file = f"{species_name}/{species_name}_rank_change_sankey.html"
    fig.write_html(out_file)

    print(f"Saved Sankey → {out_file}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    for sp, (posf, negf) in SPECIES.items():
        make_rank_change_sankey(sp, posf, negf)
'''
'''
import os
from collections import Counter
import plotly.graph_objects as go

# =============================
# CONFIGURATION
# =============================
K = 10          # Top-K k-mers per class
KMER_SIZE = 3

SPECIES = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}

# =============================
# FASTA-LIKE READER
# =============================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, "r") as f:
        seq_id, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = seq.upper()
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences[seq_id] = seq.upper()
    return sequences


# =============================
# K-MER EXTRACTION
# =============================
def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# =============================
# K-MER FREQUENCY
# =============================
def kmer_frequency(seqs, k):
    counter = Counter()
    for seq in seqs.values():
        counter.update(get_kmers(seq, k))
    return counter


# =============================
# BUILD CLASS → KMER SANKEY
# =============================
def make_class_kmer_sankey(species, pos_file, neg_file):

    print(f"\nProcessing {species}...")

    # Read sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # Count k-mers
    pos_counts = kmer_frequency(pos_seqs, KMER_SIZE)
    neg_counts = kmer_frequency(neg_seqs, KMER_SIZE)

    # Select Top-K per class
    pos_top = pos_counts.most_common(K)
    neg_top = neg_counts.most_common(K)

    # Normalize INSIDE top-K (probability mass contribution)
    pos_total = sum(v for _, v in pos_top)
    neg_total = sum(v for _, v in neg_top)

    pos_contrib = {k: v / pos_total for k, v in pos_top}
    neg_contrib = {k: v / neg_total for k, v in neg_top}

    # Nodes
    nodes = ["Positive", "Negative"]

    pos_nodes = [f"P_{k}" for k in pos_contrib]
    neg_nodes = [f"N_{k}" for k in neg_contrib]

    nodes.extend(pos_nodes)
    nodes.extend(neg_nodes)

    node_index = {n: i for i, n in enumerate(nodes)}

    # Links
    sources, targets, values, labels = [], [], [], []

    # Positive → k-mers
    for kmer, val in pos_contrib.items():
        sources.append(node_index["Positive"])
        targets.append(node_index[f"P_{kmer}"])
        values.append(val)
        labels.append(kmer)

    # Negative → k-mers
    for kmer, val in neg_contrib.items():
        sources.append(node_index["Negative"])
        targets.append(node_index[f"N_{kmer}"])
        values.append(val)
        labels.append(kmer)

    # Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels
        )
    ))

    fig.update_layout(
        title=f"{species}: Class → Top-{K} k-mer Contribution Sankey",
        font_size=11
    )

    # Save
    os.makedirs(species, exist_ok=True)
    out_file = f"{species}/{species}_class_kmer_sankey.html"
    fig.write_html(out_file)

    print(f"Saved → {out_file}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    for sp, (posf, negf) in SPECIES.items():
        make_class_kmer_sankey(sp, posf, negf)
'''
'''
import os
from collections import Counter
import plotly.graph_objects as go

# =============================
# CONFIGURATION
# =============================
K = 10          # Top-K k-mers per class
KMER_SIZE = 3

SPECIES = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}

# =============================
# FASTA-LIKE READER
# =============================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, "r") as f:
        seq_id, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = seq.upper()
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences[seq_id] = seq.upper()
    return sequences


# =============================
# K-MER EXTRACTION
# =============================
def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# =============================
# K-MER FREQUENCY
# =============================
def kmer_frequency(seqs, k):
    counter = Counter()
    for seq in seqs.values():
        counter.update(get_kmers(seq, k))
    return counter


# =============================
# MERGED-NODE SANKEY
# =============================
def make_merged_kmer_sankey(species, pos_file, neg_file):

    print(f"\nProcessing {species}...")

    # Read sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # Count k-mers
    pos_counts = kmer_frequency(pos_seqs, KMER_SIZE)
    neg_counts = kmer_frequency(neg_seqs, KMER_SIZE)

    # Top-K per class
    pos_top = dict(pos_counts.most_common(K))
    neg_top = dict(neg_counts.most_common(K))

    # Union of Top-K k-mers
    union_kmers = sorted(set(pos_top.keys()) | set(neg_top.keys()))

    # Normalize INSIDE Top-K (per class)
    pos_total = sum(pos_top.values())
    neg_total = sum(neg_top.values())

    pos_contrib = {k: pos_top[k] / pos_total for k in pos_top}
    neg_contrib = {k: neg_top[k] / neg_total for k in neg_top}

    # -------------------------
    # Nodes
    # -------------------------
    nodes = ["Positive", "Negative"] + union_kmers
    node_index = {n: i for i, n in enumerate(nodes)}

    # -------------------------
    # Links
    # -------------------------
    sources, targets, values, labels = [], [], [], []

    for kmer in union_kmers:

        # Positive → k-mer
        if kmer in pos_contrib:
            sources.append(node_index["Positive"])
            targets.append(node_index[kmer])
            values.append(pos_contrib[kmer])
            labels.append(f"Pos → {kmer}")

        # Negative → k-mer
        if kmer in neg_contrib:
            sources.append(node_index["Negative"])
            targets.append(node_index[kmer])
            values.append(neg_contrib[kmer])
            labels.append(f"Neg → {kmer}")

    # -------------------------
    # Sankey Plot
    # -------------------------
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels
        )
    ))

    fig.update_layout(
        title=f"{species}: Merged k-mer Contribution Sankey (Top-{K})",
        font_size=11
    )

    # Save
    os.makedirs(species, exist_ok=True)
    out_file = f"{species}/{species}_merged_kmer_sankey.html"
    fig.write_html(out_file)

    print(f"Saved → {out_file}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    for sp, (posf, negf) in SPECIES.items():
        make_merged_kmer_sankey(sp, posf, negf)
'''
'''
import os
from collections import Counter
import plotly.graph_objects as go

# =============================
# CONFIGURATION
# =============================
K = 10
KMER_SIZE = 3

SPECIES = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}

# =============================
# FASTA READER
# =============================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, "r") as f:
        seq_id, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = seq.upper()
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences[seq_id] = seq.upper()
    return sequences


# =============================
# K-MERS
# =============================
def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def kmer_frequency(seqs, k):
    counter = Counter()
    for seq in seqs.values():
        counter.update(get_kmers(seq, k))
    return counter


# =============================
# BIDIRECTIONAL SANKEY
# =============================
def make_bidirectional_sankey(species, pos_file, neg_file):

    print(f"\nProcessing {species}...")

    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    pos_counts = kmer_frequency(pos_seqs, KMER_SIZE)
    neg_counts = kmer_frequency(neg_seqs, KMER_SIZE)

    # Top-K k-mers in each class
    pos_top = dict(pos_counts.most_common(K))
    neg_top = dict(neg_counts.most_common(K))

    # Union of all k-mers involved
    union_kmers = sorted(set(pos_top) | set(neg_top))

    # ---- Cross contributions ----
    # Positive top-K → Negative
    pos_in_neg_total = sum(neg_counts.get(k, 0) for k in pos_top)
    pos_in_neg = {
        k: neg_counts.get(k, 0) / pos_in_neg_total
        for k in pos_top
        if pos_in_neg_total > 0
    }

    # Negative top-K → Positive
    neg_in_pos_total = sum(pos_counts.get(k, 0) for k in neg_top)
    neg_in_pos = {
        k: pos_counts.get(k, 0) / neg_in_pos_total
        for k in neg_top
        if neg_in_pos_total > 0
    }

    # ---- Nodes ----
    nodes = ["Positive", "Negative"] + union_kmers
    node_index = {n: i for i, n in enumerate(nodes)}

    # ---- Links ----
    sources, targets, values, labels = [], [], [], []

    # Positive → k-mers (measured in Negative)
    for kmer, val in pos_in_neg.items():
        sources.append(node_index["Positive"])
        targets.append(node_index[kmer])
        values.append(val)
        labels.append(f"PosTop → NegFreq: {kmer}")

    # Negative → k-mers (measured in Positive)
    for kmer, val in neg_in_pos.items():
        sources.append(node_index["Negative"])
        targets.append(node_index[kmer])
        values.append(val)
        labels.append(f"NegTop → PosFreq: {kmer}")

    # ---- Plot ----
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=labels
        )
    ))

    fig.update_layout(
        title=f"{species}: Bidirectional Top-{K} k-mer Cross-Contribution Sankey",
        font_size=11
    )

    os.makedirs(species, exist_ok=True)
    out_file = f"{species}/{species}_bidirectional_kmer_sankey.html"
    fig.write_html(out_file)

    print(f"Saved → {out_file}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    for sp, (posf, negf) in SPECIES.items():
        make_bidirectional_sankey(sp, posf, negf)
'''

'''
import os
from collections import Counter
import plotly.graph_objects as go

# =============================
# CONFIG
# =============================
K = 10
KMER_SIZE = 3

SPECIES = {
    "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
    "Human_negative_samples":("Human/neg_left_sequences.txt","Human/neg_right_sequences.txt"),
    "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
    "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
}

# =============================
# FASTA READER
# =============================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, "r") as f:
        seq_id, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = seq.upper()
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences[seq_id] = seq.upper()
    return sequences


# =============================
# K-MERS
# =============================
def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def kmer_counts(seqs, k):
    counter = Counter()
    for seq in seqs.values():
        counter.update(get_kmers(seq, k))
    return counter


# =============================
# SANKEY
# =============================
def make_20node_sankey(species, pos_file, neg_file):

    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    pos_counts = kmer_counts(pos_seqs, KMER_SIZE)
    neg_counts = kmer_counts(neg_seqs, KMER_SIZE)

    # Top-10 from each class
    pos_top = dict(pos_counts.most_common(K))
    neg_top = dict(neg_counts.most_common(K))

    # Totals for normalization
    total_pos = sum(pos_counts.values())
    total_neg = sum(neg_counts.values())

    # -----------------------------
    # Nodes
    # -----------------------------
    left_nodes = ["Positive", "Negative"]

    right_nodes = (
        [f"{k} (PosTop)" for k in pos_top.keys()] +
        [f"{k} (NegTop)" for k in neg_top.keys()]
    )

    nodes = left_nodes + right_nodes
    node_idx = {n: i for i, n in enumerate(nodes)}

    # -----------------------------
    # Links
    # -----------------------------
    sources, targets, values = [], [], []

    # For PosTop k-mers
    for kmer in pos_top:
        node_name = f"{kmer} (PosTop)"

        # Positive → k-mer
        sources.append(node_idx["Positive"])
        targets.append(node_idx[node_name])
        values.append(pos_counts[kmer] / total_pos)

        # Negative → k-mer (even if not top in negative)
        sources.append(node_idx["Negative"])
        targets.append(node_idx[node_name])
        values.append(neg_counts.get(kmer, 0) / total_neg)

    # For NegTop k-mers
    for kmer in neg_top:
        node_name = f"{kmer} (NegTop)"

        # Negative → k-mer
        sources.append(node_idx["Negative"])
        targets.append(node_idx[node_name])
        values.append(neg_counts[kmer] / total_neg)

        # Positive → k-mer (even if not top in positive)
        sources.append(node_idx["Positive"])
        targets.append(node_idx[node_name])
        values.append(pos_counts.get(kmer, 0) / total_pos)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

    fig.update_layout(
        title=f"{species}: Top-10 k-mers with Cross-Class Contribution (20-node Sankey)",
        font_size=11
    )

    os.makedirs(species, exist_ok=True)
    out = f"{species}/{species}_20node_kmer_sankey.html"
    fig.write_html(out)

    print(f"Saved: {out}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    for sp, (pf, nf) in SPECIES.items():
        make_20node_sankey(sp, pf, nf)
'''

import os
from collections import Counter
import plotly.graph_objects as go

# =============================
# CONFIG
# =============================
K = 10
KMER_SIZE = 3

# Each entry = ONE FIGURE
COMPARISONS = {
    "Human_Pos_vs_Neg": {
        "file_A": "Datasets/Human_posi_samples.txt",
        "file_B": "Datasets/Human_nega_samples.txt",
        "label_A": "Human Positive",
        "label_B": "Human Negative",
    },

    "Human_NegLeft_vs_NegRight": {
        "file_A": "Human/neg_left_sequences.txt",
        "file_B": "Human/neg_right_sequences.txt",
        "label_A": "Human Neg-Left",
        "label_B": "Human Neg-Right",
    },

    "Human_Pos_vs_NegRight": {
        "file_A": "Datasets/Human_posi_samples.txt",
        "file_B": "Human/neg_right_sequences.txt",
        "label_A": "Human Positive",
        "label_B": "Human Neg-Right",
    },

    "Mouse_Pos_vs_Neg": {
        "file_A": "Datasets/Mouse_posi_samples.txt",
        "file_B": "Datasets/Mouse_nega_samples.txt",
        "label_A": "Mouse Positive",
        "label_B": "Mouse Negative",
    },

    "Drosophila_Pos_vs_Neg": {
        "file_A": "Datasets/Drosophila_posi_samples.txt",
        "file_B": "Datasets/Drosophila_nega_samples.txt",
        "label_A": "Drosophila Positive",
        "label_B": "Drosophila Negative",
    },
}

# =============================
# FASTA READER
# =============================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, "r") as f:
        seq_id, seq = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = seq.upper()
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id:
            sequences[seq_id] = seq.upper()
    return sequences


# =============================
# K-MERS
# =============================
def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def kmer_counts(seqs, k):
    counter = Counter()
    for seq in seqs.values():
        counter.update(get_kmers(seq, k))
    return counter


# =============================
# SANKEY FUNCTION
# =============================
'''def make_20node_sankey(
    name,
    file_A,
    file_B,
    label_A,
    label_B,
):

    seqs_A = read_fasta_txt(file_A)
    seqs_B = read_fasta_txt(file_B)

    counts_A = kmer_counts(seqs_A, KMER_SIZE)
    counts_B = kmer_counts(seqs_B, KMER_SIZE)

    top_A = dict(counts_A.most_common(K))
    top_B = dict(counts_B.most_common(K))
    #print(top_A)
    #print(top_B)

    total_A = sum(counts_A.values())
    total_B = sum(counts_B.values())

    # -----------------------------
    # Nodes
    # -----------------------------
    left_nodes = [label_A, label_B]

    right_nodes = (
        [f"{k} ({label_A} Top)" for k in top_A.keys()] +
        [f"{k} ({label_B} Top)" for k in top_B.keys()]
    )

    nodes = left_nodes + right_nodes
    node_idx = {n: i for i, n in enumerate(nodes)}

    # -----------------------------
    # Links
    # -----------------------------
    sources, targets, values = [], [], []

    # Top-k from A
    for kmer in top_A:
        node = f"{kmer} ({label_A} Top)"

        sources.append(node_idx[label_A])
        targets.append(node_idx[node])
        values.append(counts_A[kmer] / total_A)

        sources.append(node_idx[label_B])
        targets.append(node_idx[node])
        values.append(counts_B.get(kmer, 0) / total_B)

    # Top-k from B
    for kmer in top_B:
        node = f"{kmer} ({label_B} Top)"

        sources.append(node_idx[label_B])
        targets.append(node_idx[node])
        values.append(counts_B[kmer] / total_B)

        sources.append(node_idx[label_A])
        targets.append(node_idx[node])
        values.append(counts_A.get(kmer, 0) / total_A)
    print(targets)
    print(sources)
    print(values)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=10,
            thickness=18,
            label=nodes
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

    fig.update_layout(
        title=f"{name}: Top-10 k-mers",
        font_size=11
    )

    os.makedirs("Sankey_Output", exist_ok=True)
    out = f"Sankey_Output/{name}_20node_kmer_sankey.html"
    fig.write_html(out)

    print(f"Saved: {out}")
'''

def make_20node_sankey(
    name,
    file_A,
    file_B,
    label_A,
    label_B,
):

    seqs_A = read_fasta_txt(file_A)
    seqs_B = read_fasta_txt(file_B)

    counts_A = kmer_counts(seqs_A, KMER_SIZE)
    counts_B = kmer_counts(seqs_B, KMER_SIZE)

    top_A = dict(counts_A.most_common(K))
    top_B = dict(counts_B.most_common(K))

    total_A = sum(counts_A.values())
    total_B = sum(counts_B.values())

    # =============================
    # UNIQUE + COMMON K-MERS
    # =============================
    set_A = set(top_A.keys())
    set_B = set(top_B.keys())

    common_kmers = set_A & set_B
    only_A = set_A - common_kmers
    only_B = set_B - common_kmers

    # -----------------------------
    # Nodes
    # -----------------------------
    left_nodes = [label_A, label_B]

    right_nodes = (
    [f"{k} ({label_A})" for k in only_A] +
    [f"{k} ({label_B})" for k in only_B] +
    [f"{k} (Common)" for k in common_kmers]
    )

    nodes = left_nodes + right_nodes
    node_idx = {n: i for i, n in enumerate(nodes)}

    # -----------------------------
    # Node colors
    # -----------------------------
    node_colors = []
    for n in nodes:
        if "(A only)" in n:
            node_colors.append("rgba(31,119,180,0.9)")   # Blue
        elif "(B only)" in n:
            node_colors.append("rgba(214,39,40,0.9)")    # Red
        elif "(Common)" in n:
            node_colors.append("rgba(44,160,44,0.9)")    # Green
        else:
            node_colors.append("rgba(160,160,160,0.8)")  # Class nodes

    # -----------------------------
    # Links
    # -----------------------------
    sources, targets, values, link_colors = [], [], [], []

    def add_link(src, tgt, val, color):
        if val > 0:
            sources.append(src)
            targets.append(tgt)
            values.append(val)
            link_colors.append(color)

    # A-only k-mers
    for kmer in only_A:
        node = f"{kmer} ({label_A})"
        color = "rgba(31,119,180,0.5)"
        add_link(node_idx[label_A], node_idx[node], counts_A[kmer]/total_A, color)
        add_link(node_idx[label_B], node_idx[node], counts_B.get(kmer, 0)/total_B, color)

    # B-only k-mers
    for kmer in only_B:
        node = f"{kmer} ({label_B})"
        color = "rgba(214,39,40,0.5)"
        add_link(node_idx[label_B], node_idx[node], counts_B[kmer]/total_B, color)
        add_link(node_idx[label_A], node_idx[node], counts_A.get(kmer, 0)/total_A, color)

    # Common k-mers
    for kmer in common_kmers:
        node = f"{kmer} (Common)"
        color = "rgba(44,160,44,0.5)"
        add_link(node_idx[label_A], node_idx[node], counts_A[kmer]/total_A, color)
        add_link(node_idx[label_B], node_idx[node], counts_B[kmer]/total_B, color)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=10,
            thickness=18,
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    ))

    fig.update_layout(
        title=f"{name}: Top-10 k-mers (Unique + Common)",
        font_size=11
    )

    os.makedirs("Sankey_Output", exist_ok=True)
    out = f"Sankey_Output/{name}_20node_kmer_sankey.html"
    fig.write_html(out)

    print(f"Saved: {out}")


# =============================
# MAIN
# =============================
if __name__ == "__main__":
    for name, cfg in COMPARISONS.items():
        make_20node_sankey(
            name=name,
            file_A=cfg["file_A"],
            file_B=cfg["file_B"],
            label_A=cfg["label_A"],
            label_B=cfg["label_B"],
        )
