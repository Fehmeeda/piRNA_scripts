import numpy as np

# Load the two npz files
arr_npz = np.load('all_3mer_embeddings.npz', allow_pickle=True)
arr_npy = np.load('embedding_matrix.npy', allow_pickle=True)


# If the npz has only one array inside
npz_arrays = arr_npz.files
if len(npz_arrays) != 1:
    print("The npz file has more than one array, cannot compare directly")
else:
    arr_from_npz = arr_npz[npz_arrays[0]]
    # Compare arrays
    if np.array_equal(arr_npy, arr_from_npz):
        print("The npy and npz arrays are exactly the same")
    else:
        print("The npy and npz arrays differ")

# List all arrays in npz
print("Arrays in npz:", arr_npz.files)
print(arr_npy[1])

# Suppose you want to compare with 'X_en_tra'
arr_from_npz = arr_npz['embeddings']
print(arr_from_npz[0])
# Compare

print("Exact equality:", np.array_equal(arr_from_npz[0], arr_npy[1]))
# Floating-point tolerant check
print("Close enough:", np.allclose(arr_from_npz[0], arr_npy[1]))
if np.array_equal(arr_npy[1], arr_from_npz[0]):
    print("Arrays are exactly the same")
else:
    print("Arrays differ")