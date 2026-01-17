import numpy as np
from itertools import product
import os
from pirna import read_fasta_txt, pad_sequences

# ==========================================
# Load dna2vec embedding (65 × 100 matrix)
# ==========================================
data = np.load("all_3mer_embeddings_with_null.npz")
embedding_matrix = data["embeddings"]   # shape: (65, 100)

# ==========================================
# Build dictionary: NULL=0, AAA=1..TTT=64
# ==========================================
def build_kmer_dict(k=3):
    bases = ["A","C","G","T"]
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    kmer_to_index = {"NULL": 0}
    for i, kmer in enumerate(kmers, start=1):
        kmer_to_index[kmer] = i
    return kmer_to_index

kmer_to_index = build_kmer_dict(k=3)

# ==========================================
# Generate disjoint k-mers with shift
# ==========================================
def get_disjoint_kmers(seq, k=3, shift=0):
    kmers = []
    for i in range(shift, len(seq) - k + 1, k):
        kmer = seq[i:i+k]
        kmers.append(kmer)
    return kmers

# ==========================================
# Convert list of k-mers → embedding matrix
# ==========================================
def kmers_to_embedding_matrix(kmer_list):
    emb_list = []
    for kmer in kmer_list:
        # If contains unknown → NULL bucket
        if "N" in kmer or kmer not in kmer_to_index:
            idx = 0
        else:
            idx = kmer_to_index[kmer]
        emb_list.append(embedding_matrix[idx])
    return np.array(emb_list)  # (#kmers, 100)

# ==========================================
# MAIN PROCESSING FUNCTION
# ==========================================
def process_species_disjoint(species_name, pos_file, neg_file, k=3):

    print(f"\n===============================")
    print(f"  Processing {species_name}")
    print(f"===============================")

    # Load sequences
    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # Pad all sequences to same length (with N)
    all_seqs = {**pos_seqs, **neg_seqs}
    padded_seqs, max_len_nt = pad_sequences(all_seqs)

    print(f"Max padded length for {species_name}: {max_len_nt}")

    # For storing:
    # X_shift0_pos, X_shift1_pos, X_shift2_pos
    # X_shift0_neg, X_shift1_neg, X_shift2_neg
    X_pos = {0: [], 1: [], 2: []}
    X_neg = {0: [], 1: [], 2: []}

    # -------- Process sequences --------
    for seq_id, seq in padded_seqs.items():

        is_pos = seq_id in pos_seqs

        for shift in [0, 1, 2]:

            kmers = get_disjoint_kmers(seq, k=k, shift=shift)
            emb = kmers_to_embedding_matrix(kmers)

            if is_pos:
                X_pos[shift].append(emb)
            else:
                X_neg[shift].append(emb)

    # Convert lists → arrays with object dtype (because lengths vary)
    #print(X_neg[0].type)
    X_pos_np = {s: np.array(X_pos[s], dtype=object) for s in [0,1,2]}
    X_neg_np = {s: np.array(X_neg[s], dtype=object) for s in [0,1,2]}
    print(X_pos_np.shape)

    # Save
    os.makedirs(species_name, exist_ok=True)

    np.savez_compressed(
        f"{species_name}/{species_name}_dna2vec_disjoint_shifts.npz",
        X_pos_shift0 = X_pos_np[0],
        X_pos_shift1 = X_pos_np[1],
        X_pos_shift2 = X_pos_np[2],
        X_neg_shift0 = X_neg_np[0],
        X_neg_shift1 = X_neg_np[1],
        X_neg_shift2 = X_neg_np[2]
    )

    print(f"Saved: {species_name}_dna2vec_disjoint_shifts.npz")


# ==========================================
# RUN FOR ALL SPECIES
# ==========================================
if __name__ == "__main__":
    species = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
    }

    for sp, (posf, negf) in species.items():
        process_species_disjoint(sp, posf, negf)
