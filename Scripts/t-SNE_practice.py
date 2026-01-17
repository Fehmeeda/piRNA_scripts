'''import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


# =============================
# Load saved data
# =============================
onehot_data = np.load('pirna_k-mer_onehot_encoding.npz', allow_pickle=True)
X_onehot = onehot_data['X']
y = onehot_data['y']
print(X_onehot.shape)

# =============================
# Flatten matrices
# =============================
# Flatten each one-hot encoding and adjacency matrix into a 1D vector
X_onehot_flat = np.array([x.flatten() for x in X_onehot])
print(X_onehot_flat.shape)

#svd = TruncatedSVD(n_components=50, random_state=42)
#X_svd = svd.fit_transform(X_onehot_flat)
#pca = PCA(n_components=50)
#X_pca = pca.fit_transform(X_onehot_flat)

tsne = TSNE (n_components=2, perplexity=30.0, early_exaggeration=12.0, 
             learning_rate='auto',max_iter=1000, n_iter_without_progress=300, 
             min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='pca', 
             verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None)

X_tsne = tsne.fit_transform(X_onehot_flat)
# =============================
# t-SNE for One-hot Encodings
# =============================

#tsne_onehot = TSNE(n_components=2, perplexity=100, random_state=42)
#X_tsne_onehot = tsne_onehot.fit_transform(X_tsne)
#print(X_tsne_onehot.shape)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='blue', label='Positive', alpha=0.6)
plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='red', label='Negative', alpha=0.6)
plt.title("t-SNE Visualization of One-hot Encoded Sequences")
plt.legend()
plt.show()
'''
# =============================
# t-SNE for Adjacency Matrices
# =============================
'''svd_graph = TruncatedSVD(n_components=50, random_state=42)
X_svd_graph = svd_graph.fit_transform(X_graph_flat)
tsne_graph = TSNE(n_components=2, perplexity=100, random_state=42)
X_tsne_graph = tsne_graph.fit_transform(X_svd_graph)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne_graph[y==1, 0], X_tsne_graph[y==1, 1], c='green', label='Positive', alpha=0.6)
plt.scatter(X_tsne_graph[y==0, 0], X_tsne_graph[y==0, 1], c='orange', label='Negative', alpha=0.6)
plt.title("t-SNE Visualization of Adjacency Matrices")
plt.legend()
plt.show()'''
'''
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# =============================
# List the files you want to load
# =============================
species_files = {
    "Human": "Human/Human_kmer_onehot_overlapping.npz",
    "Mouse": "Mouse/Mouse_kmer_onehot_overlapping.npz",
    "Drosophila": "Drosophila/Drosophila_kmer_onehot_overlapping.npz"
}

# =============================
# Process each species separately
# =============================
for species, file_path in species_files.items():
    print(f"\n=== Processing {species} ===")
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    X = data['X']  # shape: (num_sequences, 125, kmer)
    y = data['y']
    
    print(f"Loaded {species}: X={X.shape}, y={y.shape}")

    # Flatten one-hot matrices
    X_flat = np.array([x.flatten() for x in X])

    print(f"Flattened shape: {X_flat.shape}")

    # Run t-SNE
    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_flat)

    # Plot positive vs negative
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='blue', label='Positive', alpha=0.6)
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='red', label='Negative', alpha=0.6)
    plt.title(f"t-SNE of One-hot Encoded Sequences (Overlapping) — {species}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{species}/tSNE_onehot_overlapping_{species}.png")
    plt.show()
'''
import numpy as np
from itertools import product
import os
from pirna import read_fasta_txt, pad_sequences

# ==========================================
# Load dna2vec embedding (65 × 100 matrix)
# ==========================================
data = np.load("all_3mer_embeddings_with_null.npz")
embedding_matrix = data["embeddings"]         # shape: (65, 100)
NULL_EMB = embedding_matrix[0]                # 100-dim NULL embedding

# ==========================================
# Build dictionary: NULL=0, AAA=1..TTT=64
# ==========================================
def build_kmer_dict(k=3):
    bases = ["A", "C", "G", "T"]
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
        kmers.append(seq[i:i+k])
    return kmers

# ==========================================
# Convert list of k-mers → embedding matrix
# ==========================================
def kmers_to_embedding_matrix(kmer_list):
    emb_list = []
    for kmer in kmer_list:
        if "N" in kmer or kmer not in kmer_to_index:
            idx = 0
        else:
            idx = kmer_to_index[kmer]
        emb_list.append(embedding_matrix[idx])
    return np.array(emb_list)      # shape (num_kmers, 100)

# ==========================================
# Pad embedding matrices to fixed length
# ==========================================
def pad_embedding_matrix(emb_matrix, max_len):
    L = emb_matrix.shape[0]
    if L == max_len:
        return emb_matrix
    pad_count = max_len - L
    padding = np.tile(NULL_EMB, (pad_count, 1))
    return np.vstack([emb_matrix, padding])

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

    # Pad sequences to same nt length
    all_seqs = {**pos_seqs, **neg_seqs}
    padded_seqs, max_len_nt = pad_sequences(all_seqs)
    print(f"Max padded length for {species_name}: {max_len_nt}")

    # For 3 shifts
    X_pos = {0: [], 1: [], 2: []}
    X_neg = {0: [], 1: [], 2: []}

    # -------- Process all sequences --------
    for seq_id, seq in padded_seqs.items():
        is_pos = seq_id in pos_seqs

        for shift in [0, 1, 2]:
            kmers = get_disjoint_kmers(seq, k=k, shift=shift)
            emb = kmers_to_embedding_matrix(kmers)

            if is_pos:
                X_pos[shift].append(emb)
            else:
                X_neg[shift].append(emb)

    # ======================================
    # Compute global max k-mer length across all shifts
    # ======================================
    all_embs = []
    for s in [0,1,2]:
        all_embs.extend(X_pos[s])
        all_embs.extend(X_neg[s])
    global_max_kmers = max(emb.shape[0] for emb in all_embs)
    print(f"Global max kmers (all shifts): {global_max_kmers}")

    # ======================================
    # PAD ALL EMBEDDING MATRICES TO GLOBAL MAX
    # ======================================
    for s in [0, 1, 2]:
        X_pos[s] = [pad_embedding_matrix(emb, global_max_kmers) for emb in X_pos[s]]
        X_neg[s] = [pad_embedding_matrix(emb, global_max_kmers) for emb in X_neg[s]]

    # ======================================
    # CONVERT TO REAL 3D ARRAYS (NO OBJECT)
    # ======================================
    X_pos_np = {s: np.stack(X_pos[s], axis=0) for s in [0,1,2]}
    X_neg_np = {s: np.stack(X_neg[s], axis=0) for s in [0,1,2]}

    # Print shapes
    for s in [0,1,2]:
        print(f"Shift {s}: POS shape = {X_pos_np[s].shape}, NEG shape = {X_neg_np[s].shape}")

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
