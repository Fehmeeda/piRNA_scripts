import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from pirna import read_fasta_txt, pad_sequences, get_kmers,generate_kmer_dict

# =============================
# Step 2: Generate 3-mers with custom rules
# =============================
def generate_valid_kmers(k=3, alphabet='ACGTN'):
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    valid_kmers = []

    for kmer in all_kmers:

        if kmer == 'N' * k:       # Allow NNN fully
            valid_kmers.append(kmer)
            continue

        if kmer[0] == 'N':       # Skip if first is N
            continue

        if kmer[:2] == 'NN':     # Skip if first two NN
            continue

        if k >= 3 and kmer[1] == 'N' and kmer[2] != 'N' and kmer[0] != 'N':
            continue

        valid_kmers.append(kmer)

    return valid_kmers  

# =============================
# Step 4: Weighted one-hot encode k-mers
# =============================

def weighted_one_hot_kmers(kmer_list, kmer_to_index, weight_scheme=None):
    if weight_scheme is None:
        weight_scheme = {0:1, 1:0.5, 2:0.25}
    
    vec_len = len(kmer_to_index)
    seq_len = len(kmer_list)
    
    mat = np.zeros((vec_len, seq_len), dtype=float)
    
    for i, kmer in enumerate(kmer_list):
        #print(i,kmer)
        for dist, weight in weight_scheme.items():
            # self or previous neighbor
            if i - dist >= 0:
            
                idx_prev = kmer_to_index.get(kmer_list[i - dist])
                if idx_prev is not None:
                    mat[idx_prev, i] += weight
                    #print(i)
                
            # next neighbor (skip self)
            if dist != 0 and i + dist < seq_len:
                idx_next = kmer_to_index.get(kmer_list[i + dist])
                if idx_next is not None:
                    mat[idx_next, i] += weight
                    #print(i)
    return mat


'''

def weighted_one_hot_kmers_with_motif(kmer_list, kmer_to_index, weight_scheme=None):
    if weight_scheme is None:
        weight_scheme = {0:1, 1:0.5, 2:0.25}
    
    vec_len = len(kmer_to_index)
    seq_len = len(kmer_list)
    mat = np.zeros((vec_len, seq_len), dtype=float)
    
    # Regular weighted one-hot
    for i, kmer in enumerate(kmer_list):
        for dist, weight in weight_scheme.items():
            if i - dist >= 0:
                idx_prev = kmer_to_index.get(kmer_list[i - dist])
                if idx_prev is not None:
                    mat[idx_prev, i] += weight
            if dist != 0 and i + dist < seq_len:
                idx_next = kmer_to_index.get(kmer_list[i + dist])
                if idx_next is not None:
                    mat[idx_next, i] += weight

    # ===== Add motif-specific weights =====
    motif_weight = 1.0  # boost weight for motifs

    # 1. First T
    if kmer_list[0].startswith("T"):
        # boost first 5 positions
        for i in range(min(5, seq_len)):
            idx = kmer_to_index.get(kmer_list[i])
            if idx is not None:
                mat[idx, i] += motif_weight

    # 2. 10A rule
    if seq_len >= 10 and kmer_list[9].startswith("A"):  # 0-based index
        for i in range(max(0, 5), min(seq_len, 14)):
            idx = kmer_to_index.get(kmer_list[i])
            if idx is not None:
                mat[idx, i] += motif_weight

    return mat

'''
'''
def weighted_one_hot_kmers_with_motif(kmer_list, kmer_to_index, weight_scheme=None):
    if weight_scheme is None:
        weight_scheme = {0: 1, 1: 0.5, 2: 0.25}

    vec_len = len(kmer_to_index)
    seq_len = len(kmer_list)
    mat = np.zeros((vec_len, seq_len), dtype=float)

    # ===============================
    # 1. Regular weighted one-hot
    # ===============================
    for i, kmer in enumerate(kmer_list):
        for dist, weight in weight_scheme.items():
            # backward
            if i - dist >= 0:
                idx_prev = kmer_to_index.get(kmer_list[i - dist])
                if idx_prev is not None:
                    mat[idx_prev, i] += weight

            # forward (skip dist=0)
            if dist != 0 and i + dist < seq_len:
                idx_next = kmer_to_index.get(kmer_list[i + dist])
                if idx_next is not None:
                    mat[idx_next, i] += weight

    # ===============================
    # 2. Motif-specific weights
    # ===============================
    motif_weight = 1.0

    # -------- Rule A: First k-mer starts with "T" --------
    if seq_len > 0 and kmer_list[0].startswith("T"):
        idx0 = kmer_to_index.get(kmer_list[0])
        #print(idx0,kmer_list[0])
        if idx0 is not None:
            mat[idx0, 0] += motif_weight

    # -------- Rule B: 10th nucleotide motif (index 9) --------
    # Apply only if k-mers cover that position
    if seq_len > 9:

        # 10th kmer (index 9)
        tenth_kmer = kmer_list[9]

        # Check motif: kmer starts with A
        if tenth_kmer.startswith("A"):

            # ---- Boost the 10th kmer itself ----
            idx10 = kmer_to_index.get(tenth_kmer)
            if idx10 is not None:
                mat[idx10, 9] += motif_weight

            # ---- Boost all kmers whose window includes sequence position 10 ----
            # For 3-mers: positions {8, 9, 10}
            
            target_positions = [8, 10]

            for pos in target_positions:
                if 0 <= pos < seq_len:
                    kx = kmer_list[pos]
                    idxp = kmer_to_index.get(kx)
                    if idxp is not None:
                        mat[idxp, pos] += motif_weight

    return mat
'''
# =============================
# Step 5: Weighted adjacency matrix
# =============================
def adjacency_weighted(kmer_list, kmer_to_index, weight_scheme=None):
    weighted_oh = weighted_one_hot_kmers(kmer_list, kmer_to_index, weight_scheme)
    return np.dot(weighted_oh, weighted_oh.T)

# =============================
# Step 6: Process one species
# =============================
def process_species(species_name, pos_file, neg_file):
    print(f"\n===============================\nProcessing {species_name}\n===============================")

    # Read sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)
    all_seqs = {**pos_seqs, **neg_seqs}
    
    # Pad sequences globally
    padded_all, max_len = pad_sequences(all_seqs)
    print(f"Padding {species_name} sequences to length: {max_len}")

    # Prepare directories
    os.makedirs(f"{species_name}_pos_images_weighted", exist_ok=True)
    os.makedirs(f"{species_name}_neg_images_weighted", exist_ok=True)
    os.makedirs(f"{species_name}", exist_ok=True)

    # Generate valid kmer dictionary
    valid_kmers = generate_valid_kmers(k=3)
    kmer_to_index = {kmer: i for i, kmer in enumerate(valid_kmers)}
   
    print(f"Number of valid kmers: {len(valid_kmers)}")

    # Storage
    X_list, y_list, adj_list = [], [], []

    for i, (seq_id, seq) in enumerate(padded_all.items()):
        kmers = get_kmers(seq, k=3)
        weighted_oh = weighted_one_hot_kmers(kmers, kmer_to_index)
        X_list.append(weighted_oh)
        #print(weighted_oh.shape)
        

        if i == 0:
            print(f"\nWeighted one-hot matrix for sequence {seq_id}:")
            np.set_printoptions(precision=2, suppress=True)
            print(weighted_oh)
            np.savetxt(f"{species_name}_weighted_matrix.csv", weighted_oh, delimiter=",", fmt="%.4f")

        # Determine label
        label = 1 if seq_id in pos_seqs else 0
        y_list.append(label)

        # Save image
        folder = f"{species_name}_pos_images_weighted" if label == 1 else f"{species_name}_neg_images_weighted"
        plt.imsave(os.path.join(folder, f"{seq_id}.png"), weighted_oh, cmap="gray")

        # Weighted adjacency
        adj_list.append(adjacency_weighted(kmers, kmer_to_index))

    # Save arrays
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    adj = np.stack(adj_list, axis=0)

    np.savez_compressed(f"{species_name}/{species_name}_kmer_weighted_onehot.npz", X=X, y=y)
    np.savez_compressed(f"{species_name}/{species_name}_graph_weighted.npz", adj=adj, y=y)

    print(f"Saved: {species_name}_kmer_weighted_onehot.npz")
    print(f"Saved: {species_name}_graph_weighted.npz")

    # Visual comparison (10 pos vs 10 neg)
    pos_imgs = [os.path.join(f"{species_name}_pos_images_weighted", f) for f in os.listdir(f"{species_name}_pos_images_weighted") if f.endswith(".png")]
    neg_imgs = [os.path.join(f"{species_name}_neg_images_weighted", f) for f in os.listdir(f"{species_name}_neg_images_weighted") if f.endswith(".png")]

    pos_samples = random.sample(pos_imgs, min(10, len(pos_imgs)))
    neg_samples = random.sample(neg_imgs, min(10, len(neg_imgs)))

    plt.figure(figsize=(15, 6))
    for idx, p in enumerate(pos_samples):
        plt.subplot(2, 10, idx + 1)
        plt.imshow(Image.open(p).convert("L"), cmap="gray")
        plt.axis("off")
        plt.title("Pos", fontsize=8)
    for idx, p in enumerate(neg_samples):
        plt.subplot(2, 10, idx + 11)
        plt.imshow(Image.open(p).convert("L"), cmap="gray")
        plt.axis("off")
        plt.title("Neg", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{species_name}/{species_name}_comparison_weighted.png", dpi=300)
    plt.close()
    print(f"Saved comparison: {species_name}_comparison_weighted.png")

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
