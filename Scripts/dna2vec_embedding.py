import numpy as np
from itertools import product
import os
from pirna import read_fasta_txt, pad_sequences, get_kmers

# ===============================
# Load prebuilt 3-mer embedding
# ===============================
data = np.load("all_3mer_embeddings_with_null.npz")
embedding_matrix = data["embeddings"]   # shape: (65, 100)
print(embedding_matrix)

# ===============================
# Build dictionary: NULL=0, AAA=1, ... TTT=64
# ===============================
def build_kmer_dict(k=3):
    bases = ["A","C","G","T"]
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    kmer_to_index = {"NULL": 0}
    for i, kmer in enumerate(kmers, start=1):
        kmer_to_index[kmer] = i
    return kmer_to_index

kmer_to_index = build_kmer_dict(k=3)

# ===============================
# Sequence → dna2vec embedding with NULL handling
# ===============================
def sequence_to_embedding_matrix(seq, k=3):
    kmers = get_kmers(seq, k)
    emb_list = []
    for kmer in kmers:
        if "N" in kmer or kmer not in kmer_to_index:
            idx = 0
        else:
            idx = kmer_to_index[kmer]
        emb_list.append(embedding_matrix[idx])
    return np.array(emb_list)   # shape = (#kmers, 100)

# ===============================
# Process one species
# ===============================
def process_species(species_name, pos_file, neg_file, k=3):
    print(f"Processing {species_name}...")

    pos_seqs = read_fasta_txt(pos_file)
    neg_seqs = read_fasta_txt(neg_file)

    # Combine all sequences and pad with 'N' at the end
    all_seqs = {**pos_seqs, **neg_seqs}
    
    padded_seqs, max_len_nt = pad_sequences(all_seqs)
    print(f"Max sequence length (nucleotides) for {species_name}: {max_len_nt}")

    X_list = []
    y_list = []

    for seq_id, seq in padded_seqs.items():
        emb = sequence_to_embedding_matrix(seq, k=k)
        X_list.append(emb)
        label = 1 if seq_id in pos_seqs else 0
        y_list.append(label)
    
    
    # Convert to array (num_sequences, max_kmers, 100)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    os.makedirs(species_name, exist_ok=True)
    np.savez_compressed(f"{species_name}/{species_name}_dna2vec_embeddings.npz", X=X, y=y)
    print(f"Saved {species_name}_dna2vec_embeddings.npz with shape {X.shape}")

# ===============================
# MAIN – run for all species
# ===============================
if __name__ == "__main__":
    species = {
        "Human": ("Datasets/Human_posi_samples.txt", "Datasets/Human_nega_samples.txt"),
        "Mouse": ("Datasets/Mouse_posi_samples.txt", "Datasets/Mouse_nega_samples.txt"),
        "Drosophila": ("Datasets/Drosophila_posi_samples.txt", "Datasets/Drosophila_nega_samples.txt"),
    }

    for sp, (posf, negf) in species.items():
        process_species(sp, posf, negf)
