import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# =============================
# Step 0: Read sequences
# =============================
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


# =============================
# Step 1: Pad sequences
# =============================
def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences.values())
    padded = {sid: seq + 'N'*(max_len - len(seq)) for sid, seq in sequences.items()}
    return padded, max_len


# =============================
# Step 2: Generate 3-mers
# =============================
def get_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# =============================
# Step 3: K-mer dictionary (125)
# =============================
def generate_kmer_dict(k=3, alphabet='ACGTN'):
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


# =============================
# Step 4: One-hot encode k-mers
# (125 X kmer)
# =============================
def one_hot_encode_kmers(kmer_list, kmer_to_index):
    vec_len = len(kmer_to_index)
    mat = np.zeros((vec_len,len(kmer_list)), dtype=int)
    for i, kmer in enumerate(kmer_list):
        idx = kmer_to_index.get(kmer)
        if idx is not None:
            mat[idx, i] = 1
    return mat

# =============================
# Step 5: Adjacency matrix (125 x 125)
# =============================
def adjacency(kmer_list, kmer_to_index):
    one_hot = one_hot_encode_kmers(kmer_list, kmer_to_index)
    X = one_hot  # shape 125 x kmer
    #print(np.dot(X,X.T))
    return np.dot(X, X.T)


# =============================
# PROCESS ONE SPECIES
# =============================
def process_species(species_name, pos_file, neg_file):

    print(f"\n===============================")
    print(f"   Processing {species_name}")
    print(f"===============================")

    # Read files
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # ------------ Find global padding ------------
    all_seqs = {**pos_seqs, **neg_seqs}
    padded_all, max_len = pad_sequences(all_seqs)
    print(f"Padding {species_name} sequences to length: {max_len}")

    # prepare dirs
    os.makedirs(f"{species_name}_pos_images_overlapping", exist_ok=True)
    os.makedirs(f"{species_name}_neg_images_overlapping", exist_ok=True)

    # kmer dict
    kmer_to_index = generate_kmer_dict(k=3)

    # storage
    X_list = []
    y_list = []
    adj_list = []

    # ------------ Process each sequence ------------
    for seq_id, seq in padded_all.items():

        kmers = get_kmers(seq, k=3)

        # One-hot matrix (125 X len_kmer)
        oh = one_hot_encode_kmers(kmers, kmer_to_index)
        X_list.append(oh) 

        # Determine label
        label = 1 if seq_id in pos_seqs else 0
        y_list.append(label)

        # Save image
        folder = f"{species_name}_pos_images_overlapping" if label == 1 else f"{species_name}_neg_images_overlapping"
        img_path = os.path.join(folder, f"{seq_id}.png")
        plt.imsave(img_path, oh, cmap="gray")

        # Adjacency
        adj_list.append(adjacency(kmers, kmer_to_index))
    

    # ------------ Save arrays ------------
    X = np.stack(X_list, axis=0)
    #print(f"Shape of X is {X.shape}")
    y = np.array(y_list)
    adj = np.stack(adj_list, axis=0)  

    np.savez_compressed(f"{species_name}/{species_name}_kmer_onehot_overlapping.npz", X=X, y=y)
    np.savez_compressed(f"{species_name}/{species_name}_graph_overlapping.npz", adj=adj, y=y)

    print(f"Saved: {species_name}_kmer_onehot_overlapping.npz")
    print(f"Saved: {species_name}_graph_overlapping.npz")

    # ------------ Visual comparison (10 pos vs 10 neg) ------------
    pos_imgs = [os.path.join(f"{species_name}_pos_images_overlapping", f)
                for f in os.listdir(f"{species_name}_pos_images_overlapping")
                if f.endswith(".png")]

    neg_imgs = [os.path.join(f"{species_name}_neg_images_overlapping", f)
                for f in os.listdir(f"{species_name}_neg_images_overlapping")
                if f.endswith(".png")]

    pos_samples = random.sample(pos_imgs, min(10, len(pos_imgs)))
    neg_samples = random.sample(neg_imgs, min(10, len(neg_imgs)))

    plt.figure(figsize=(15, 6))

    # top row
    for i, p in enumerate(pos_samples):
        plt.subplot(2, 10, i + 1)
        plt.imshow(Image.open(p).convert("L"), cmap="gray")
        plt.axis("off")
        plt.title("Pos", fontsize=8)

    # bottom row
    for i, p in enumerate(neg_samples):
        plt.subplot(2, 10, i + 11)
        plt.imshow(Image.open(p).convert("L"), cmap="gray")
        plt.axis("off")
        plt.title("Neg", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{species_name}/{species_name}_comparison.png", dpi=300)
    plt.close()

    print(f"Saved comparison: {species_name}_comparison.png")


# =============================
# MAIN: RUN FOR ALL SPECIES
# =============================
if __name__ == "__main__":

    species = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
    }

    for sp, (posf, negf) in species.items():
        process_species(sp, posf, negf)