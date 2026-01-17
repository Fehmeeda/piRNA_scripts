import numpy as np
from itertools import product
import os
from pirna import read_fasta_txt, pad_sequences   # your existing functions

# ==================================
# 1. Load dna2vec embedding matrix
# ==================================
data = np.load("all_3mer_embeddings_with_null.npz")
embedding_matrix = data["embeddings"]   # shape: (65, 100)

# ==================================
# 2. Build 3-mer index dictionary
# ==================================
def build_kmer_dict(k=3):
    bases = ["A","C","G","T"]
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    kmer_to_index = {"NULL": 0}
    for i, kmer in enumerate(kmers, start=1):
        kmer_to_index[kmer] = i
    return kmer_to_index

kmer_to_index = build_kmer_dict(k=3)

# ==================================
# 3. Get *DISJOINT* k-mers (non-overlapping)
# ==================================
def get_disjoint_kmers(seq, k=3):
    kmers = []
    for i in range(0, len(seq), k):   # STEP = k (no overlap)
        chunk = seq[i:i+k]
        if len(chunk) < k:
            chunk = "NULL"
        kmers.append(chunk)
    return kmers

# ==================================
# 4. Convert sequence → disjoint dna2vec embedding
# ==================================
def sequence_to_embedding_matrix_disjoint(seq, k=3):
    kmers = get_disjoint_kmers(seq, k)

    emb_list = []
    for kmer in kmers:
        if ("N" in kmer) or (kmer not in kmer_to_index):
            idx = 0
        else:
            idx = kmer_to_index[kmer]
        emb_list.append(embedding_matrix[idx])

    return np.array(emb_list)   # shape = (num_kmers_disjoint, 100)

# ==================================
# 5. Process one species
# ==================================
def process_species_disjoint(species_name, pos_file, neg_file, k=3):
    print(f"\nProcessing {species_name} (DISJOINT kmers)...")

    # Load FASTA
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # Combine dictionaries
    all_seqs = {**pos_seqs, **neg_seqs}

    # Pad sequences to same length *in nucleotides*
    padded_seqs, max_len_nt = pad_sequences(all_seqs)

    # compute number of disjoint k-mers
    max_disjoint_kmers = int(np.ceil(max_len_nt / k))
    print(f"Max disjoint k-mers = {max_disjoint_kmers}")

    X_list = []
    y_list = []

    for seq_id, seq in padded_seqs.items():
        emb = sequence_to_embedding_matrix_disjoint(seq, k)
        
        # pad DNA2vec (if needed)
        if emb.shape[0] < max_disjoint_kmers:
            pad_len = max_disjoint_kmers - emb.shape[0]
            pad_block = np.zeros((pad_len, embedding_matrix.shape[1]))
            emb = np.vstack([emb, pad_block])

        X_list.append(emb)
        y_list.append(1 if seq_id in pos_seqs else 0)

    # Convert
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    # Save
    os.makedirs(species_name, exist_ok=True)
    np.savez_compressed(
        f"{species_name}/{species_name}_dna2vec_disjoint.npz",
        X=X,
        y=y
    )

    print(f"Saved {species_name}_dna2vec_disjoint.npz → X shape = {X.shape}")

# ==================================
# 6. MAIN
# ==================================
if __name__ == "__main__":
    species = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
    }

    for sp, (posf, negf) in species.items():
        process_species_disjoint(sp, posf, negf, k=3)
