import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from pirna import pad_sequences,read_fasta_txt, get_kmers

# =============================
# Step 3: Generate 85 valid kmers
# =============================
def generate_valid_kmers(k=3, alphabet='ACGTN'):
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    valid_kmers = []
    for kmer in all_kmers:
        if kmer == 'NNN':           # allow full N
            valid_kmers.append(kmer)
            continue
        if kmer[0] == 'N':          # skip kmers starting with N
            continue
        if kmer[:2] == 'NN':        # skip kmers with first 2 N
            continue
        if k >= 3 and kmer[1] == 'N' and kmer[2] != 'N' and kmer[0] != 'N':
            continue
        valid_kmers.append(kmer)
    return valid_kmers

# =============================
# Step 4: One-hot encode kmers
# =============================
def one_hot_encode_kmers(kmer_list, kmer_to_index):
    vec_len = len(kmer_to_index)
    mat = np.zeros((vec_len, len(kmer_list)), dtype=int)
    for i, kmer in enumerate(kmer_list):
        idx = kmer_to_index.get(kmer)
        if idx is not None:  # skip invalid kmers automatically
            mat[idx, i] = 1
    return mat

# =============================
# Step 5: Adjacency matrix
# =============================
def adjacency(kmer_list, kmer_to_index):
    one_hot = one_hot_encode_kmers(kmer_list, kmer_to_index)
    return np.dot(one_hot, one_hot.T)

# =============================
# Step 6: Process one species
# =============================
def process_species(species_name, pos_file, neg_file):
    print(f"\n===============================\nProcessing {species_name}\n===============================")

    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)
    all_seqs = {**pos_seqs, **neg_seqs}

    padded_all, max_len = pad_sequences(all_seqs)
    print(f"Padding {species_name} sequences to length: {max_len}")

    os.makedirs(f"{species_name}_pos_images", exist_ok=True)
    os.makedirs(f"{species_name}_neg_images", exist_ok=True)

    valid_kmers = generate_valid_kmers(k=3)
    kmer_to_index = {kmer: i for i, kmer in enumerate(valid_kmers)}
    print(f"Number of valid kmers: {len(valid_kmers)}")  # should be 85

    X_list, y_list, adj_list = [], [], []

    for seq_id, seq in padded_all.items():
        kmers = get_kmers(seq, k=3)
        oh = one_hot_encode_kmers(kmers, kmer_to_index)
        X_list.append(oh)

        label = 1 if seq_id in pos_seqs else 0
        y_list.append(label)

        folder = f"{species_name}_pos_images" if label == 1 else f"{species_name}_neg_images"
        plt.imsave(os.path.join(folder, f"{seq_id}.png"), oh, cmap="gray")

        adj_list.append(adjacency(kmers, kmer_to_index))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    adj = np.stack(adj_list, axis=0)

    np.savez_compressed(f"{species_name}/{species_name}_kmer_onehot_85.npz", X=X, y=y)
    np.savez_compressed(f"{species_name}/{species_name}_graph_85.npz", adj=adj, y=y)

    print(f"Saved: {species_name}_kmer_onehot_85.npz")
    print(f"Saved: {species_name}_graph_85.npz")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    species = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
    }

    for sp, (posf, negf) in species.items():
        process_species(sp, posf, negf)
