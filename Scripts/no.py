import numpy as np

data = np.load(
    "Drosophila/Drosophila_kmer_weighted_onehot.npz",
    allow_pickle=True
)

X = data["X"]
y = data["y"]

# Remove print truncation
np.set_printoptions(threshold=np.inf)

# Transpose the first sample
X0_T = X[0].T

print("Original X[0] shape:", X[0].shape)
print("Transposed X[0] shape:", X0_T.shape)

print("\nFULL X[0] (columns first):")
print(X0_T)

print("\nLabel:", y[0])
