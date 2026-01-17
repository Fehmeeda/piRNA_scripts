import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# ============================
# Flatten for t-SNE
# ============================
def flatten_embeddings(emb_array):
    # emb_array: (#seq, max_len, 100)
    #X_flat = np.array([x.flatten() for x in X])
    #print(f"Flattened shape: {X_flat.shape}")
    #####
    #####
    ####check this

    emb_array= np.array([x.flatten() for x in emb_array])
    print(emb_array.shape)
    return emb_array

# ============================
# Run t-SNE for POS vs NEG
# ============================
def run_tsne_pos_vs_neg(pos_array, neg_array, species_name, pos_shift, neg_shift):
    """
    pos_array, neg_array: (#seq, max_len, 100)
    """
    # Combine
    X_comb = np.vstack([pos_array, neg_array])
    y_comb = np.array([1]*pos_array.shape[0] + [0]*neg_array.shape[0])

    # Flatten for t-SNE
    X_flat = flatten_embeddings(X_comb)

    # Run t-SNE
    tsne = TSNE()
    X_tsne = tsne.fit_transform(X_flat)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[y_comb==1,0], X_tsne[y_comb==1,1], c='blue', alpha=0.6, label='POS')
    plt.scatter(X_tsne[y_comb==0,0], X_tsne[y_comb==0,1], c='red', alpha=0.6, label=f'NEG shift {neg_shift}')
    plt.legend()
    plt.title(f"{species_name}: POS shift {pos_shift} vs NEG shift {neg_shift}")
    plt.tight_layout()

    # Save
    out_dir = species_name
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/tSNE_POSshift{pos_shift}_vs_NEGshift{neg_shift}.png", dpi=300)
    plt.show()
    plt.close()
    print(f"Saved: {out_dir}/tSNE_POSshift{pos_shift}_vs_NEGshift{neg_shift}.png")

# ============================
# MAIN: iterate over species and shifts
# ============================
if __name__ == "__main__":
    species_files = {
        "Human": "Human/Human_dna2vec_disjoint_shifts.npz",
        "Mouse": "Mouse/Mouse_dna2vec_disjoint_shifts.npz",
        "Drosophila": "Drosophila/Drosophila_dna2vec_disjoint_shifts.npz"
    }

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
