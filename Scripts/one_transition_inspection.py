'''import numpy as np
import itertools

K = 3
ALPHABET = ["A", "C", "G", "T"]
MATRIX_PATH = "output_state_matrices/Human/overlapping/positive/positive_4.npy"

KMERS = ["".join(p) for p in itertools.product(ALPHABET, repeat=K)]

# SHOW EVERYTHING
np.set_printoptions(threshold=np.inf, linewidth=300)

mat = np.load(MATRIX_PATH)

print("Matrix shape:", mat.shape)
print("Total transitions:", mat.sum())
print("\nFULL MATRIX:\n")
print(mat)'''


import numpy as np
import itertools

K = 3
ALPHABET = ["A", "C", "G", "T"]
MATRIX_PATH = "output_state_matrices/Human/overlapping/negative/negative_15.npy"

KMERS = ["".join(p) for p in itertools.product(ALPHABET, repeat=K)]

mat = np.load(MATRIX_PATH)

print("Matrix shape:", mat.shape)
print("Total transitions:", mat.sum())
print("\nNon-zero transitions:\n")

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if mat[i, j] > 0:
            print(f"{KMERS[i]} → {KMERS[j]} : {int(mat[i, j])}")

