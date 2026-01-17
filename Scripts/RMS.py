import numpy as np
import csv
import itertools

# =========================
# 1. Load embedding matrix
# =========================
data = np.load("all_3mer_embeddings_with_null.npz")
embedding_matrix = data['embeddings']   # shape: (65, 100)

# =========================
# 2. Recreate kmer_to_index
# =========================
bases = ["A", "C", "G", "T"]
k = 3

kmer_to_index = {"NULL": 0}
all_kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]

for idx, kmer in enumerate(all_kmers, start=1):
    kmer_to_index[kmer] = idx

# =========================
# 3. Compute RMS per k-mer
# =========================
# RMS formula: sqrt(mean(v_i^2))
rms_per_kmer = np.sqrt(np.mean(np.square(embedding_matrix), axis=1))

# =========================
# 4. Print RMS values
# =========================
print("RMS values of k-mers:")
for kmer, idx in kmer_to_index.items():
    print(f"{kmer}: {rms_per_kmer[idx]:.6f}")

# =========================
# 5. Save to CSV
# =========================
with open("kmer_rms_values.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k-mer", "RMS Amplitude"])
    for kmer, idx in kmer_to_index.items():
        writer.writerow([kmer, rms_per_kmer[idx]])

print("\nSaved RMS values to kmer_rms_values.csv")

'''import pandas as pd
import numpy as np
import itertools

# Your kmers
bases = ["A", "C", "G", "T"]
all_kmers = ["".join(p) for p in itertools.product(bases, repeat=3)]

# Make sure embedding_matrix matches the order of all_kmers
# embedding_matrix.shape -> (len(all_kmers)+1, embedding_dim) if you included "NULL" at index 0
# Skip the NULL row if present
embedding_matrix_no_null = embedding_matrix[1:]  # skip first row if it's "NULL"

# Create DataFrame
df = pd.DataFrame(embedding_matrix_no_null, index=all_kmers)
df.index.name = "kmer"

# Save to CSV
df.to_csv("embedding_matrix_with_kmers.csv")
print("Saved embedding matrix with k-mers as CSV!")
'''