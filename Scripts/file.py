'''import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- Load data ---
onehot_data = np.load('pirna_k-mer_onehot_encoding.npz', allow_pickle=True)
X = onehot_data['X']
y = onehot_data['y']
#print(y)
print("Original shape:", X.shape)

# --- Flatten 3D to 2D (required for t-SNE) ---
#X_flat = X.reshape(X.shape[0], -1)
X_flat = np.array([x.flatten() for x in X])
print("Flattened shape:", X_flat.shape)

# --- Step 1: Optional PCA (recommended before t-SNE) ---
#pca = PCA(n_components=50, random_state=42)
#X_pca = pca.fit_transform(X_flat)
#print("After PCA:", X_pca.shape)
#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD(n_components=5, random_state=10)
#X_svd = svd.fit_transform(X_flat)

# --- Step 2: Apply t-SNE ---
tsne = TSNE(n_components=2, perplexity=10, random_state=10)
X_tsne = tsne.fit_transform(X_flat)
print("After t-SNE:", X_tsne.shape)
print(X_tsne)
# --- Step 3: Plot ---
print(X_tsne[y==1,1])
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], color='blue', label='Negative')
plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], color='red', label='Positive')

plt.legend()
plt.title("t-SNE Visualization of 3-mer One-Hot Encoded Sequences")
plt.show()'''
'''
from sklearn.manifold import TSNE
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_flat)
tsne = TSNE(n_components=3, perplexity=20, random_state=20)
X_tsne = tsne.fit_transform(X_svd)   # X = your feature matrix
'''
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Positive samples (y == 1)
ax.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], X_tsne[y==1, 2],
           color='red', label='Positive', s=10, alpha=0.6)

# Negative samples (y == 0)
ax.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], X_tsne[y==0, 2],
           color='blue', label='Negative', s=10, alpha=0.6)

ax.set_title("3D t-SNE Visualization of 3-mer One-Hot Encoded Sequences")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
ax.legend()
plt.show()'''
''''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# --- Load dna2vec features ---
data = np.load('pirna_k-mer_onehot_encoding.npz', allow_pickle=True)
X = data['X']
y = data['y']

print("Loaded X:", X.shape)

# --- Flatten if necessary ---
if len(X.shape) > 2:
    X_flat = X.reshape(X.shape[0], -1)
else:
    X_flat = X

# --- Dimensionality reduction before t-SNE ---
#svd = TruncatedSVD(n_components=50, random_state=42)
#X_svd = svd.fit_transform(X_flat)

# --- Settings ---
perplexities = [5, 20, 40]           # Try a few perplexities
random_states = [10, 20,30]          # Try a few random seeds

# --- Plot grid ---
fig, axes = plt.subplots(len(perplexities), len(random_states),
                         figsize=(15, 12))

for i, perp in enumerate(perplexities):
    for j, rs in enumerate(random_states):
        tsne = TSNE(n_components=2, perplexity=perp,
                    learning_rate=10, 
                    random_state=rs)
        X_tsne = tsne.fit_transform(X_flat)

        ax = axes[i, j]
        ax.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1],
                   color='blue', label='Negative', alpha=0.6, s=10)
        ax.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1],
                   color='red', label='Positive', alpha=0.6, s=10)
        ax.set_title(f"Perp={perp}, Random={rs}")
        ax.set_xticks([]); ax.set_yticks([])

# --- Legend and layout ---
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.show()'''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

# --- Load dna2vec features ---
data = np.load('pirna_k-mer_onehot_encoding.npz', allow_pickle=True)
X = data['X']
y = data['y']

print("Loaded X:", X.shape)

# --- Flatten if necessary ---
if len(X.shape) > 2:
    X_flat = X.reshape(X.shape[0], -1)
else:
    X_flat = X

# --- Dimensionality reduction before t-SNE ---
#svd = TruncatedSVD(n_components=50, random_state=42)
#X_svd = svd.fit_transform(X_flat)

# --- Random seeds from your friend ---
random_seeds = [
    7, 13, 21, 42, 77, 88, 99, 
    123, 256, 321, 512, 731, 
    941, 1151, 2024, 2161, 4096, 9001
]

# --- Create folder to save results ---
save_dir = "tsne_results_one_hot_encoding"
os.makedirs(save_dir, exist_ok=True)

# --- Generate and save t-SNE plots ---
for seed in random_seeds:
    print(f"Running t-SNE with random seed {seed} ...")

    tsne = TSNE(
        n_components=2,
        max_iter=3000,
        init='random',
        perplexity=20,
        random_state=seed,
        learning_rate=200
    )
    X_tsne = tsne.fit_transform(X_flat)

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1],
                color='blue', label='Negative', alpha=0.6, s=10)
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1],
                color='red', label='Positive', alpha=0.6, s=10)
    plt.title(f"t-SNE (Perplexity=20, Random Seed={seed})")
    plt.legend()
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    # --- Save each image ---
    save_path = os.path.join(save_dir, f"tsne_seed_{seed}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

print("\n✅ All t-SNE images saved in folder:", save_dir)