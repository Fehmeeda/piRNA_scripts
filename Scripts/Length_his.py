# ===============================
# Sequence Length Analysis:
# Positive vs Negative Samples
# For Human, Mouse, Drosophila
# ===============================

import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
from pirna import read_fasta_txt

# ============================
# Process One Species
# ============================
def process_species(pos_file, neg_file, species_name):
    print(f"\n==============================")
    print(f"Processing {species_name}")
    print(f"==============================")

    # Load sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # Compute lengths
    pos_lengths = [len(seq) for seq in pos_seqs.values()]
    neg_lengths = [len(seq) for seq in neg_seqs.values()]

    # Print stats
    print(f"{species_name} Positive: min={min(pos_lengths)}, max={max(pos_lengths)}, total={len(pos_lengths)}")
    print(f"{species_name} Negative: min={min(neg_lengths)}, max={max(neg_lengths)}, total={len(neg_lengths)}")

    # ----- Build tables -----
    pos_counts = Counter(pos_lengths)
    neg_counts = Counter(neg_lengths)

    # All unique lengths combined for consistent x-axis
    all_lengths = sorted(set(pos_counts.keys()) | set(neg_counts.keys()))

    # Frequency lists for plotting
    pos_freq = [pos_counts.get(l, 0) for l in all_lengths]
    neg_freq = [neg_counts.get(l, 0) for l in all_lengths]

    # ----- Save CSVs -----
    pos_df = pd.DataFrame({"sequence_length": all_lengths, "count": pos_freq})
    neg_df = pd.DataFrame({"sequence_length": all_lengths, "count": neg_freq})

    pos_df.to_csv(f"{species_name}/{species_name}_positive_length_distribution.csv", index=False)
    neg_df.to_csv(f"{species_name}/{species_name}_negative_length_distribution.csv", index=False)
    print(f"Saved CSVs for {species_name}")

    # ----- Plot combined histogram -----
    x = np.arange(len(all_lengths))  # positions for bars
    width = 0.4  # width of bars

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, pos_freq, width=width, color='skyblue', edgecolor='black', label='Positive')
    plt.bar(x + width/2, neg_freq, width=width, color='salmon', edgecolor='black', label='Negative')

    plt.xticks(x, all_lengths, rotation=90)
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title(f"{species_name}: Positive vs Negative Sequence Lengths")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{species_name}/{species_name}_length_histogram_pos_vs_neg.png", dpi=300)
    plt.show()
    print(f"Saved combined histogram for {species_name}")

# ============================
# RUN FOR ALL SPECIES
# ============================
if __name__ == "__main__":

    # Human
    process_species(
        pos_file="Datasets/Human_posi_samples.txt",
        neg_file="Datasets/Human_nega_samples.txt",
        species_name="Human"
    )

    # Mouse
    process_species(
        pos_file="Datasets/Mouse_posi_samples.txt",
        neg_file="Datasets/Mouse_nega_samples.txt",
        species_name="Mouse"
    )

    # Drosophila
    process_species(
        pos_file="Datasets/Drosophila_posi_samples.txt",
        neg_file="Datasets/Drosophila_nega_samples.txt",
        species_name="Drosophila"
    )
