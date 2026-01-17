import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pirna import read_fasta_txt, pad_sequences, generate_kmer_dict, get_kmers
import os

def process_species_kmer_freq(pos_file, neg_file, species_name):
    
    # Read sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)
    all_seqs = {**pos_seqs, **neg_seqs}
    print(f"{species_name}: Total sequences = {len(all_seqs)}")
    
    # Pad sequences
    padded_seqs, max_len = pad_sequences(all_seqs)
    print(f"{species_name}: Padded sequences to length {max_len}")
    
    # Generate all 125 3-mers
    kmer_to_index = generate_kmer_dict(k=3)
    all_kmers = list(kmer_to_index.keys())
    
    # Count k-mers in all sequences
    kmer_counter = Counter()
    for seq in padded_seqs.values():
        kmers = get_kmers(seq, k=3)
        kmer_counter.update(kmers)
    
    # Create frequency DataFrame
    freq_list = [kmer_counter.get(k, 0) for k in all_kmers]
    df_freq = pd.DataFrame({"kmer": all_kmers, "frequency": freq_list})
    
    # Save CSV
    csv_path = os.path.join(f"{species}/overlapping_kmer_frequency_table_{species_name}.csv")
    df_freq.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Plot histogram
    plt.figure(figsize=(20,6))
    plt.bar(all_kmers, freq_list, color='skyblue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.xlabel("3-mer")
    plt.ylabel("Frequency (padded sequences)")
    plt.title(f"{species_name}: Frequency of All 125 3-mers")
    plt.tight_layout()
    plt.savefig(os.path.join(f"{species_name}/overlapping_kmer_frequency_barplot_{species_name}.png"), dpi=300)
    plt.show()
    print(f"Saved barplot for {species_name}")
    
    # Print zero-count kmers
    missing = [k for k, v in zip(all_kmers, freq_list) if v == 0]
    print(f"{species_name}: {len(missing)} kmers with zero count")
    if missing:
        print("Examples (first 20):", missing[:20])
    
    return df_freq

# =========================
# Main
# =========================
if __name__ == "__main__":
    species_files = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt")
    }
    
    for species, (pos_file, neg_file) in species_files.items():
        process_species_kmer_freq(pos_file, neg_file, species)
