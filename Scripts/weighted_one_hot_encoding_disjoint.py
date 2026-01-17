import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from pirna import read_fasta_txt, pad_sequences


# ============================================================
# 1. DISJOINT KMER GENERATION
# ============================================================
'''def get_disjoint_kmers(seq, k=3):
    """Return non-overlapping disjoint kmers."""
    return [seq[i:i+k] for i in range(0, len(seq), k) if len(seq[i:i+k]) == k]'''
def get_disjoint_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq), k):
        chunk = seq[i:i+k]
        if len(chunk) < k:
            chunk += 'N'*(k - len(chunk))
        kmers.append(chunk)
    return kmers

# ============================================================
# 2. VALID KMER GENERATOR (with your special N-handling rules)
# ============================================================
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


# ============================================================
# 3. Weighted one-hot encode **disjoint** kmers
# ============================================================
def weighted_one_hot_kmers(kmer_list, kmer_to_index, weight_scheme=None):
    """
    Compute weighted one-hot matrix for disjoint kmers.
    weight_scheme: distance → weight
    """
    if weight_scheme is None:
        # Suggested weights: distance = 0 → central kmer, distance = 1 → next disjoint block (= k bp away)
        weight_scheme = {0: 1.0, 1: 0.5, 2: 0.25}

    vec_len = len(kmer_to_index)
    seq_len = len(kmer_list)

    mat = np.zeros((vec_len, seq_len), dtype=float)

    for i, kmer in enumerate(kmer_list):

        for dist, weight in weight_scheme.items():

            # backward neighbor
            if i - dist >= 0:
                idx_prev = kmer_to_index.get(kmer_list[i - dist])
                if idx_prev is not None:
                    mat[idx_prev, i] += weight

            # forward neighbor
            if dist != 0 and i + dist < seq_len:
                idx_next = kmer_to_index.get(kmer_list[i + dist])
                if idx_next is not None:
                    mat[idx_next, i] += weight

    return mat


# ============================================================
# 4. Adjacency matrix for disjoint kmers
# ============================================================
def adjacency_weighted(kmer_list, kmer_to_index, weight_scheme=None):
    weighted_oh = weighted_one_hot_kmers(kmer_list, kmer_to_index, weight_scheme)
    return np.dot(weighted_oh, weighted_oh.T)   # Gram matrix


# ============================================================
# 5. Process ONE species (DISJOINT ONLY)
# ============================================================
def process_species(species_name, pos_file, neg_file):
    print(f"\n===============================")
    print(f"Processing {species_name} (DISJOINT kmers)")
    print(f"===============================")

    # Read sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)
    all_seqs = {**pos_seqs, **neg_seqs}

    # Pad sequences
    padded_all, max_len = pad_sequences(all_seqs)
    print(f"Padded to length: {max_len}")

    # Make folders
    os.makedirs(f"{species_name}_pos_images_disjoint", exist_ok=True)
    os.makedirs(f"{species_name}_neg_images_disjoint", exist_ok=True)
    os.makedirs(f"{species_name}", exist_ok=True)

    # Generate valid kmers
    valid_kmers = generate_valid_kmers(k=3)
    kmer_to_index = {kmer: i for i, kmer in enumerate(valid_kmers)}

    print(f"Valid kmers: {len(valid_kmers)}")

    X_list, y_list, adj_list = [], [], []

    for i, (seq_id, seq) in enumerate(padded_all.items()):

        # ========= DISJOINT kmers here =========
        kmers = get_disjoint_kmers(seq, k=3)

        # Weighted one-hot
        weighted_oh = weighted_one_hot_kmers(kmers, kmer_to_index)
        X_list.append(weighted_oh)

        # Show & save first matrix
        if i == 0:
            np.set_printoptions(precision=2, suppress=True)
            print(f"\nWeighted disjoint one-hot for {seq_id}:")
            print(weighted_oh)
            np.savetxt(f"{species_name}/{species_name}_first_disjoint_matrix.csv",
                       weighted_oh, delimiter=",", fmt="%.4f")

        # Labeling
        label = 1 if seq_id in pos_seqs else 0
        y_list.append(label)

        # Save images
        folder = f"{species_name}_pos_images_disjoint" if label == 1 else f"{species_name}_neg_images_disjoint"
        plt.imsave(os.path.join(folder, f"{seq_id}.png"), weighted_oh, cmap="gray")

        # Adjacency
        adj_list.append(adjacency_weighted(kmers, kmer_to_index))

    # Convert to arrays
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    adj = np.stack(adj_list, axis=0)

    # Save npz datasets
    np.savez_compressed(f"{species_name}/{species_name}_kmer_weighted_disjoint_onehot.npz", X=X, y=y)
    np.savez_compressed(f"{species_name}/{species_name}_graph_weighted_disjoint.npz", adj=adj, y=y)

    print(f"Saved: {species_name}_kmer_disjoint_onehot.npz")
    print(f"Saved: {species_name}_graph_disjoint.npz")

    # Save comparison visualization
    pos_imgs = [os.path.join(f"{species_name}_pos_images_disjoint", f)
                for f in os.listdir(f"{species_name}_pos_images_disjoint") if f.endswith(".png")]
    neg_imgs = [os.path.join(f"{species_name}_neg_images_disjoint", f)
                for f in os.listdir(f"{species_name}_neg_images_disjoint") if f.endswith(".png")]

    pos_samples = random.sample(pos_imgs, min(10, len(pos_imgs)))
    neg_samples = random.sample(neg_imgs, min(10, len(neg_imgs)))

    plt.figure(figsize=(15, 6))
    for idx, p in enumerate(pos_samples):
        plt.subplot(2, 10, idx + 1)
        plt.imshow(Image.open(p).convert("L"), cmap="gray")
        plt.axis("off")

    for idx, p in enumerate(neg_samples):
        plt.subplot(2, 10, idx + 11)
        plt.imshow(Image.open(p).convert("L"), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{species_name}/{species_name}_comparison_disjoint.png", dpi=300)
    plt.close()

    print(f"Saved comparison image: {species_name}_comparison_disjoint.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    species = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
       "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
    }

    for sp, (pos_file, neg_file) in species.items():
        process_species(sp, pos_file, neg_file)
