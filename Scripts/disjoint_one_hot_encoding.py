# ===============================
# Disjoint 3-mer One-Hot Encoding
# Frequency & Histogram
# Human, Mouse, Drosophila
# ===============================
# Note: 
# Have not added Y label in the disjoint npy files

import numpy as np
from itertools import product
import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt
from pirna import read_fasta_txt, generate_kmer_dict, pad_sequences, one_hot_encode_kmers
   
# -------------------------------
# Step 1: Disjoint 3-mers
# -------------------------------
def get_disjoint_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq), k):
        chunk = seq[i:i+k]
        if len(chunk) < k:
            chunk += 'N'*(k - len(chunk))
        kmers.append(chunk)
    return kmers

# -------------------------------
# Step 2: Process one species
# -------------------------------
def process_species(pos_file, neg_file, species_name):
    print(f"\nProcessing {species_name}...")

    # Read sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    all_seqs = {**pos_seqs, **neg_seqs}
    padded_seqs, max_len = pad_sequences(all_seqs)
    print(f"Padded sequences to length: {max_len}")

    # Generate k-mer dictionary
    kmer_to_index = generate_kmer_dict(k=3)
    all_kmers = list(kmer_to_index.keys())

    # Containers
    X_pos_list = []
    X_neg_list = []
    y_list = []
    kmer_counter_pos = Counter()
    kmer_counter_neg = Counter()
    kmer_counter_combined = Counter()

    # ----- Positive samples -----
    for seq_id, seq in pos_seqs.items():
        padded_seq = padded_seqs[seq_id]
        kmers = get_disjoint_kmers(padded_seq, k=3)
        kmer_counter_pos.update(kmers)
        kmer_counter_combined.update(kmers)
        one_hot = one_hot_encode_kmers(kmers, kmer_to_index)
        X_pos_list.append(one_hot)
    print(f"Processed {len(X_pos_list)} positive sequences")
    #print(X_pos_list)

    # ----- Negative samples -----
    for seq_id, seq in neg_seqs.items():
        padded_seq = padded_seqs[seq_id]
        kmers = get_disjoint_kmers(padded_seq, k=3)
        kmer_counter_neg.update(kmers)
        kmer_counter_combined.update(kmers)
        one_hot = one_hot_encode_kmers(kmers, kmer_to_index)
        X_neg_list.append(one_hot)
    print(f"Processed {len(X_neg_list)} negative sequences")


    # Convert to numpy arrays
    X_pos = np.array(X_pos_list, dtype=object)
    X_neg = np.array(X_neg_list, dtype=object)

    # ----- Save one-hot matrices -----
    np.save(f'{species_name}/{species_name}_X_pos_disjoint.npy', X_pos)
    np.save(f'{species_name}/{species_name}_X_neg_disjoint.npy', X_neg)
    print("Saved one-hot matrices for positive and negative samples")

    # ----- Save k-mer frequency -----
    freq_list_pos = [kmer_counter_pos.get(k,0) for k in all_kmers]
    freq_list_neg = [kmer_counter_neg.get(k,0) for k in all_kmers]
    freq_list_combined = [kmer_counter_combined.get(k,0) for k in all_kmers]

    df_pos = pd.DataFrame({"kmer": all_kmers, "frequency": freq_list_pos})
    df_neg = pd.DataFrame({"kmer": all_kmers, "frequency": freq_list_neg})
    df_combined = pd.DataFrame({"kmer": all_kmers, "frequency": freq_list_combined})

    df_pos.to_csv(f'{species_name}/kmer_frequency_{species_name}_pos_disjoint.csv', index=False)
    df_neg.to_csv(f'{species_name}/kmer_frequency_{species_name}_neg_disjoint.csv', index=False)
    df_combined.to_csv(f'{species_name}/kmer_frequency_{species_name}_combined_disjoint.csv', index=False)
    print("Saved k-mer frequency tables for positive, negative, and combined samples")

    # ----- Histogram of combined frequencies -----
    plt.figure(figsize=(20,6))
    plt.bar(all_kmers, freq_list_combined, color='skyblue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.xlabel("3-mer")
    plt.ylabel("Frequency (Padded Sequences)")
    plt.title(f"{species_name}: Disjoint 3-mer Frequency")
    plt.tight_layout()
    plt.savefig(f"{species_name}/{species_name}_combined_kmer_disjoint_histogram.png", dpi=300)
    plt.show()
    print(f"Saved histogram for {species_name} combined 3-mer frequencies")

    return X_pos, X_neg, df_pos, df_neg, df_combined

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    species_files = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt")
    }

    for species, (pos_file, neg_file) in species_files.items():
        process_species(pos_file, neg_file, species)
