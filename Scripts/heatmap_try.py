'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os

from torch import neg

# ============================================================
# CONFIGURATION
# ============================================================

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

    plt.figure(figsize=(10, 18))
    ax = sns.heatmap(
        combined,
        cmap="viridis",
        yticklabels=True,
        annot=False   # set True if your professor wants numbers
    )
    # ----- separator line between POS and NEG -----
    ax.hlines(
        y=len(pos),
        xmin=0,
        xmax=combined.shape[1],
        colors="white",
        linewidth=2
    )
    # ----- group annotations -----
    ax.text(
        combined.shape[1] + 0.5,
        len(pos) / 2,
        "POS",
        va="center",
        fontsize=12,
        fontweight="bold"
    )
    ax.text(
        combined.shape[1] + 0.5,
        len(pos) + len(neg) / 2,
        "NEG",
        va="center",
        fontsize=12,
        fontweight="bold"
    )
    plt.title(
        f"{species} ({mode})\n"
        "Position-Specific K-mer Frequency\n"
        "Column → Row Normalized"
    )
    plt.xlabel("Position")
    plt.ylabel("K-mer")
    plt.tight_layout()
    plt.savefig(
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
if __name__ == "__main__":
    main()'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os

# ============================================================
# CONFIGURATION
# ============================================================
POS_COLOR = "#1f77b4"   # blue
NEG_COLOR = "#d62728"   # red

BASE_DIR = "output_position_kmer_frequency"
FIG_DIR = "heatmaps"
os.makedirs(FIG_DIR, exist_ok=True)

SPECIES = ["Drosophila", "Human", "Mouse"]
MODES = ["disjoint", "overlap"]

K = 3
ALPHABET = "ACGT"

POS_COLOR = "steelblue"
NEG_COLOR = "firebrick"

# ============================================================
# FIXED K-MER ORDER
# ============================================================

def generate_kmers(k, alphabet):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

KMER_ORDER = generate_kmers(K, ALPHABET)

# ============================================================
# STEP 1: PREPARATION NORMALIZATION (COLUMN ONLY)
# ============================================================

def load_and_prepare(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    df = df.reindex(KMER_ORDER)

    # per-position probability distribution
    df = df.div(df.sum(axis=0), axis=1)
    #print(df)

    return df

# ============================================================
# HEATMAP 1: POSITION-WISE COMPARISON
# (COLUMN NORMALIZATION AFTER COMBINING)
# ============================================================
def plot_position_heatmap(pos, neg, species, mode):

    pos_l = pos.copy()
    neg_l = neg.copy()

    # ❌ DO NOT modify index with POS / NEG
    combined = pd.concat([pos_l, neg_l], axis=0)

    # column normalization AFTER merge
    combined = combined.div(combined.sum(axis=0), axis=1)
    

    fig = plt.figure(figsize=(11, 18))

    # layout: [LEFT BAR | HEATMAP]
    gs = fig.add_gridspec(1, 2, width_ratios=[0.4, 10], wspace=0.02)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    sns.heatmap(
        combined,
        ax=ax,
        cmap="viridis",
        yticklabels=True,
        cbar=True
    )

    # separator between POS and NEG
    ax.hlines(
        y=len(pos_l),
        xmin=0,
        xmax=combined.shape[1],
        colors="white",
        linewidth=2
    )
    # move k-mers to the right
    ax.yaxis.tick_right()
    ax.set_ylabel("")

    # 🔥 FIX ROTATION
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# optional: smaller font
    ax.tick_params(axis="y", labelsize=6)

    # ===== LEFT CLASS COLOR BAR =====
    class_colors = (
        [POS_COLOR] * len(pos_l) +
        [NEG_COLOR] * len(neg_l)
    )

    color_rgb = [plt.matplotlib.colors.to_rgb(c) for c in class_colors]
    bar_array = [[c] for c in color_rgb]  # (n_rows, 1)

    ax_bar.imshow(
        bar_array,
        aspect="auto",
        interpolation="nearest",
        extent=[0, 1, 0, combined.shape[0]]
    )

    ax_bar.set_ylim(ax.get_ylim())
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_frame_on(False)   # optional: cleaner look

    plt.suptitle(
        f"{species} ({mode})\n"
        "Position-wise K-mer Distribution\n"
        "Column-normalized after POS+NEG merge",
        fontsize=14
    )

    plt.savefig(
        f"{FIG_DIR}/{species}_{mode}_position_comparison.png",
        dpi=300
    )
    plt.close()


# ============================================================
# HEATMAP 2: K-MER-WISE COMPARISON
# (ROW NORMALIZATION AFTER COMBINING)
# ============================================================
def plot_kmer_heatmap(pos, neg, species, mode):

    pos = pos.loc[KMER_ORDER]
    neg = neg.loc[KMER_ORDER]

    pos.columns = [f"{c}_POS" for c in pos.columns]
    neg.columns = [f"{c}_NEG" for c in neg.columns]

    combined = pd.concat([pos, neg], axis=1)
    

    # Row-wise normalization AFTER merge
    combined = combined.div(combined.sum(axis=1), axis=0)


    fig = plt.figure(figsize=(14, 15))

    # Grid:
    # [ TOP BAR | COLORBAR ]
    # [ HEATMAP | COLORBAR ]
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.3, 10],
        width_ratios=[20, 1],
        hspace=0.02,
        wspace=0.05
    )

    ax_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        combined,
        ax=ax,
        cmap="viridis",
        yticklabels=True,
        cbar=True,
        cbar_ax=cax
    )

    # Separator between POS and NEG
    ax.vlines(
        x=pos.shape[1],
        ymin=0,
        ymax=combined.shape[0],
        colors="white",
        linewidth=2
    )

    # ===== TOP CLASS BAR =====
    class_colors = (
        ["#1f77b4"] * pos.shape[1] +
        ["#d62728"] * neg.shape[1]
    )

    bar_rgb = [[plt.matplotlib.colors.to_rgb(c) for c in class_colors]]

    n_cols = combined.shape[1]

    ax_bar.imshow(
        bar_rgb,
        aspect="auto",
        interpolation="nearest",
        extent=[0, n_cols, 0, 1]
    )


    # 🔥 CRITICAL: exact sync
    ax_bar.set_xlim(ax.get_xlim())

    ax_bar.set_xticks([])
    ax_bar.set_yticks([])

    plt.suptitle(
        f"{species} ({mode})\n"
        "K-mer-wise Distribution Across Positions\n"
        "Row-normalized after POS+NEG merge",
        fontsize=14
    )

    plt.savefig(
        f"{FIG_DIR}/{species}_{mode}_kmer_comparison.png",
        dpi=300
    )
    plt.close()

# ============================================================
# MAIN LOOP
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

            pos = load_and_prepare(pos_file)
            neg = load_and_prepare(neg_file)

            plot_position_heatmap(pos, neg, species, mode)
            plot_kmer_heatmap(pos, neg, species, mode)

    print("✅ All heatmaps generated successfully.")

if __name__ == "__main__":
    main()
