import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# ============================
# Load disjoint shift embeddings
# ============================
species_files = {
    "Human": "Human/Human_dna2vec_disjoint_shifts.npz",
    "Mouse": "Mouse/Mouse_dna2vec_disjoint_shifts.npz",
    "Drosophila": "Drosophila/Drosophila_dna2vec_disjoint_shifts.npz"
}

# ============================
# Pad embeddings to max length
# ============================
def pad_embeddings(emb_list, max_len=None):
    """
    emb_list: list of arrays (#kmers, 100)
    returns: array (#seq, max_len, 100)
    """
    n_seqs = len(emb_list)
    feature_dim = emb_list[0].shape[1]

    if not max_len:
        max_len = max(e.shape[0] for e in emb_list)

    padded = np.zeros((n_seqs, max_len, feature_dim), dtype=float)
    for i, e in enumerate(emb_list):
        padded[i, :e.shape[0], :] = e
    return padded

# ============================
# Flatten for t-SNE
# ============================
def flatten_embeddings(emb_array):
    # emb_array: (#seq, max_len, 100)
    n_seq = emb_array.shape[0]
    return emb_array.reshape(n_seq, -1)

# ============================
# Run umap for POS vs single NEG shift
# ============================
def run_tsne_pos_vs_neg(pos_list, neg_list, species_name, pos_shift, neg_shift):
    # Pad separately to max length
    max_len = max(max(e.shape[0] for e in pos_list), max(e.shape[0] for e in neg_list))
    X_pos = pad_embeddings(pos_list, max_len)
    X_neg = pad_embeddings(neg_list, max_len)

    # Combine
    X_comb = np.vstack([X_pos, X_neg])
    y_comb = np.array([1]*len(pos_list) + [0]*len(neg_list))

    # Flatten
    X_flat = flatten_embeddings(X_comb)

    # --- UMAP instead of t-SNE ---
    import umap
    umap_model = umap.UMAP()
    X_umap = umap_model.fit_transform(X_flat)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(X_umap[y_comb==1,0], X_umap[y_comb==1,1], c='blue', alpha=0.6, label='POS')
    plt.scatter(X_umap[y_comb==0,0], X_umap[y_comb==0,1], c='red', alpha=0.6, label=f'NEG shift {neg_shift}')
    plt.legend()
    plt.title(f"{species_name}: POS shift {pos_shift} vs NEG shift {neg_shift}")
    plt.tight_layout()
    
    # Save
    out_dir = species_name
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/UMAP_POSshift{pos_shift}_vs_NEGshift{neg_shift}.png", dpi=300)
    
    plt.show()
    plt.close()
    print(f"Saved: {out_dir}/UMAP_POSshift{pos_shift}_vs_NEGshift{neg_shift}.png")


# ============================
# MAIN: iterate over species and shifts
# ============================
if __name__ == "__main__":
    for species_name, file_path in species_files.items():
        print(f"\n=== t-SNE: {species_name} ===")
        data = np.load(file_path, allow_pickle=True)
        
        # Load POS and NEG embeddings per shift
        pos_shifts = [data[f"X_pos_shift{i}"] for i in range(3)]
        neg_shifts = [data[f"X_neg_shift{i}"] for i in range(3)]
        
        # Run POS shift i vs each NEG shift j
        for pos_shift in range(3):
            for neg_shift in range(3):
                run_tsne_pos_vs_neg(pos_shifts[pos_shift], neg_shifts[neg_shift], species_name, pos_shift, neg_shift)
