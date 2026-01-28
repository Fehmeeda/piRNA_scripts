import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os

# ============================================================
# CONFIGURATION
# ============================================================
GROUP_COLORS = {
    "POS": "#d62728",  # red
    "NEG": "#1f77b4"   # blue
}

BASE_DIR = "output_position_kmer_frequency"

SPECIES = [
    "Drosophila",
    "Human",
    "Mouse"
]

MODES = [
    "disjoint",
    "overlap"
]

K = 3                  # k-mer length
ALPHABET = "ACGT"

FIG_DIR = "heatmaps"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# K-MER ORDER (CRITICAL: fixes scrambled rows)
# ============================================================

def generate_kmers(k, alphabet):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

KMER_ORDER = generate_kmers(K, ALPHABET)

# ============================================================
# LOAD + COLUMN-WISE NORMALIZATION
# ============================================================

def load_and_normalize(csv_file, kmer_order):
    df = pd.read_csv(csv_file, index_col=0)

    # enforce identical k-mer order everywhere
    df = df.reindex(kmer_order)

    # column-wise normalization
    df = df.div(df.sum(axis=0), axis=1)

    # row-wise normalization
    df = df.div(df.sum(axis=1), axis=0)

    return df

# ============================================================
# HEATMAP 1: POSITION-WISE (POS TOP, NEG BOTTOM)
# ============================================================
def plot_position_heatmap(pos, neg, species, mode):
    pos_l = pos.copy()
    pos_l.index = pos_l.index + "_POS"

    neg_l = neg.copy()
    neg_l.index = neg_l.index + "_NEG"

    combined = pd.concat([pos_l, neg_l], axis=0)

    # ----- row color annotation -----
    row_colors = [
        GROUP_COLORS["POS"] if "_POS" in idx else GROUP_COLORS["NEG"]
        for idx in combined.index
    ]

    g = sns.clustermap(
        combined,
        cmap="viridis",
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        figsize=(10, 18)
    )

    g.fig.suptitle(
        f"{species} ({mode})\n"
        "Position-Specific K-mer Frequency\n"
        "Column → Row Normalized",
        y=1.02
    )

    g.ax_heatmap.set_xlabel("Position")
    g.ax_heatmap.set_ylabel("K-mer")

    g.savefig(
        f"{FIG_DIR}/{species}_{mode}_position_comparison.png",
        dpi=300
    )
    plt.close()


def plot_kmer_heatmap(pos, neg, species, mode):
    pos = pos.loc[KMER_ORDER]
    neg = neg.loc[KMER_ORDER]

    pos.columns = [f"{c}_POS" for c in pos.columns]
    neg.columns = [f"{c}_NEG" for c in neg.columns]

    combined = pd.concat([pos, neg], axis=1)

    plt.figure(figsize=(14, 14))
    ax = sns.heatmap(
        combined,
        cmap="viridis",
        yticklabels=True,
        annot=False
    )

    # ----- separator line -----
    ax.vlines(
        x=pos.shape[1],
        ymin=0,
        ymax=combined.shape[0],
        colors="white",
        linewidth=2
    )

    # ----- group labels -----
    ax.text(
        pos.shape[1] / 2,
        -1.5,
        "POS",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )
    ax.text(
        pos.shape[1] + neg.shape[1] / 2,
        -1.5,
        "NEG",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        f"{species} ({mode})\n"
        "K-mer Distribution Across Positions\n"
        "Column → Row Normalized"
    )
    plt.xlabel("Position & Class")
    plt.ylabel("K-mer")

    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/{species}_{mode}_kmer_comparison.png",
        dpi=300
    )
    plt.close()

# ============================================================
# MAIN LOOP: ALL SPECIES × DISJOINT / OVERLAPPING
# ============================================================

def main():
    for species in SPECIES:
        for mode in MODES:

            pos_file = f"{BASE_DIR}/{species}_pos_{mode}_table.csv"
            neg_file = f"{BASE_DIR}/{species}_neg_{mode}_table.csv"

            if not os.path.exists(pos_file) or not os.path.exists(neg_file):
                print(f"[SKIP] Missing files for {species} ({mode})")
                continue

            print(f"[PROCESS] {species} ({mode})")

            pos = load_and_normalize(pos_file, KMER_ORDER)
            neg = load_and_normalize(neg_file, KMER_ORDER)

            plot_position_heatmap(pos, neg, species, mode)
            plot_kmer_heatmap(pos, neg, species, mode)

    print("✅ All heatmaps generated successfully.")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
