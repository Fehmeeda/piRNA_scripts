# ========================================================
# Kmer Frequency of Positive and negatives samples separately of each specie
# ========================================================

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


# =========================================================
# Step 0 — Read FASTA/TXT sequences
# =========================================================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, 'r') as f:
        seq_id = ''
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    sequences[seq_id] = seq
                seq_id = line[1:]
                seq = ''
            else:
                seq += line.upper()
        if seq_id:
            sequences[seq_id] = seq
    return sequences


# =========================================================
# Step 1 — Pad sequences
# =========================================================
def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences.values())

    padded = {}
    for k, seq in sequences.items():
        padded[k] = seq + 'N' * (max_len - len(seq))

    return padded, max_len


# =========================================================
# Step 2 — Generate overlapping k-mers
# =========================================================
def get_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# =========================================================
# Step 3 — Generate full k-mer dictionary
# =========================================================
def generate_kmer_dict(k=3, alphabet='ACGTN'):
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(kmers)}


# =========================================================
# MAIN FUNCTION — Compute k-mer frequency
# =========================================================
def compute_kmer_freq(seq_dict, species_name, label, k=3):
    """
    seq_dict = {id: sequence}
    species_name = "Human" / "Mouse" / "Drosophila"
    label = "positive" or "negative"
    """

    print(f"\n=== Computing k-mer frequencies for {species_name} ({label}) ===")

    # Pad sequences
    padded_seqs, max_len = pad_sequences(seq_dict)
    print(f"Padded {species_name} {label} sequences to length: {max_len}")

    # Prepare k-mer dictionary (125 3-mers)
    kmer_to_idx = generate_kmer_dict(k)
    all_kmers = list(kmer_to_idx.keys())

    # Count k-mers
    counter = Counter()
    for seq_id, seq in padded_seqs.items():
        kmers = get_kmers(seq, k)
        counter.update(kmers)

    freq_list = [counter.get(kmer, 0) for kmer in all_kmers]

    # Build DataFrame
    df = pd.DataFrame({
        "kmer": all_kmers,
        "frequency": freq_list
    })#.sort_values("kmer").reset_index(drop=True)

    # Save CSV
    csv_name = f"kmer_frequency_{species_name}_{label}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved CSV: {csv_name}")

    # Plot histogram
    plt.figure(figsize=(20, 6))
    plt.bar(all_kmers, freq_list, edgecolor="black")
    plt.xticks(rotation=90)
    plt.xlabel("3-mer")
    plt.ylabel("Frequency")
    plt.title(f"{species_name} — {label.capitalize()} Samples — 3-mer Frequency")
    plt.tight_layout()

    img_name = f"kmer_histogram_{species_name}_{label}.png"
    plt.savefig(img_name, dpi=300)
    plt.show()
    print(f"Saved histogram: {img_name}")

    # Missing kmers
    missing = [k for k, f in zip(all_kmers, freq_list) if f == 0]
    print(f"Missing 3-mers: {len(missing)}")

    return df

# =========================================================
# EXECUTE FOR ALL SPECIES & LABELS WITH SAME PADDING
# =========================================================
if __name__ == "__main__":

    species_files = {
        "Human": ("Human_posi_samples.txt", "Human_nega_samples.txt"),
        "Mouse": ("Mouse_posi_samples.txt", "Mouse_nega_samples.txt"),
        "Drosophila": ("Drosophila_posi_samples.txt", "Drosophila_nega_samples.txt"),
    }

    for species_name, (pos_file, neg_file) in species_files.items():

        print(f"\n==============================")
        print(f"Processing {species_name}")
        print(f"==============================")

        # Read
        pos = read_fasta_txt(pos_file)
        neg = read_fasta_txt(neg_file)

        # ---- Find global max len ----
        all_lengths = [len(s) for s in pos.values()] + [len(s) for s in neg.values()]
        global_max = max(all_lengths)

        print(f"Global padding length for {species_name}: {global_max}")

        # ---- Pad both with SAME length ----
        pos_padded, _ = pad_sequences(pos, max_len=global_max)
        neg_padded, _ = pad_sequences(neg, max_len=global_max)

        # ---- Compute freq using padded ----
        compute_kmer_freq(pos_padded, species_name, "positive")
        compute_kmer_freq(neg_padded, species_name, "negative")
