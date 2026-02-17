import numpy as np
import os
from itertools import product
from pirna import read_fasta_txt

# =========================
# CONFIG
# =========================
K = 3
SPECIES = ["Human", "Mouse", "Drosophila"]
FOLDS = range(5)
OUTDIR = "encoded_features"
kmertype = "overlap" # disjoint | overlap
os.makedirs(OUTDIR, exist_ok=True)

# ===============================
# Load dna2vec
# ===============================
dna2vec_data = np.load("all_3mer_embeddings_with_null.npz")
DNA2VEC_EMB = dna2vec_data["embeddings"]


# =========================
# KMER UTILS (SAME AS CNN)
# =========================
def get_kmers(seq, k, kmertype):
    if kmertype == "overlap":
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]
    else:
        return [seq[i:i+k] for i in range(0, len(seq) - k + 1, k)] 
    
#To include N kmers Just add N in alphabet, If N is not in the alphabet even the NNN is not counted as a valid kmer and have 0's vector instead
def generate_valid_kmers(k=3, alphabet="ACGTN"):
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    valid = []

    for kmer in all_kmers:
        if kmer == "N" * k:
            valid.append(kmer)
            continue
        if kmer[0] == "N":
            continue
        if kmer[:2] == "NN":
            continue
        if k >= 3 and kmer[1] == "N" and kmer[2] != "N":
            continue
        valid.append(kmer)

    return valid

#Include N in alphabet
def weighted_one_hot_kmers(kmer_list, kmer_to_index, weights={0:1.0,1:0.5,2:0.25}):
    mat = np.zeros((len(kmer_to_index), len(kmer_list)), dtype=np.float32)

    for i in range(len(kmer_list)):
        for d, w in weights.items():
            if i - d >= 0:
                idx = kmer_to_index.get(kmer_list[i - d])
                if idx is not None:
                    mat[idx, i] += w
            if d != 0 and i + d < len(kmer_list):
                idx = kmer_to_index.get(kmer_list[i + d])
                if idx is not None:
                    mat[idx, i] += w
    return mat

#✅ Do NOT include N in alphabet
#✅ Do NOT include N in kmer dictionary
#✅ Skip kmers containing N
'''def weighted_one_hot_kmers(kmer_list, kmer_to_index, weights={0:1.0,1:0.5,2:0.25}):
   mat = np.zeros((len(kmer_to_index), len(kmer_list)), dtype=np.float32)


   for i in range(len(kmer_list)):
       kmer = kmer_list[i]
       if 'N' in kmer:
           continue  # skip k-mers with N
       for d, w in weights.items():
           if i - d >= 0:
               idx = kmer_to_index.get(kmer_list[i - d])
               if idx is not None and 'N' not in kmer_list[i - d]:
                   mat[idx, i] += w
           if d != 0 and i + d < len(kmer_list):
               idx = kmer_to_index.get(kmer_list[i + d])
               if idx is not None and 'N' not in kmer_list[i + d]:
                   mat[idx, i] += w


  
   return mat'''

def dna2vec_sequence_matrix(seq, k, kmer_to_index):
    kmers = get_kmers(seq, k, kmertype)
    emb_list = []
    for kmer in kmers:
        if "N" in kmer or kmer not in kmer_to_index:
            idx = 0
        else:
            idx = kmer_to_index[kmer]
        emb_list.append(DNA2VEC_EMB[idx])
    return np.array(emb_list, dtype=np.float32)   # shape = (#kmers, 100)

def build_kmer_dict(K):
    bases = ["A","C","G","T"]
    kmers = ["".join(p) for p in product(bases, repeat=K)]
    kmer_to_index = {"NULL": 0}
    for i, kmer in enumerate(kmers, start=1):
        kmer_to_index[kmer] = i
    return kmer_to_index


# =========================
# ENCODING FUNCTION
# =========================
def encode_split(pos_file, neg_file, kmer_to_index, kmer_to_index_for_dna2vec, cnn_len):
    pos = read_fasta_txt(pos_file)
    neg = read_fasta_txt(neg_file)

    X_oh, X_d2v, y, ids = [], [], [], []

    for sid, seq in {**pos, **neg}.items():
        seq = seq[:cnn_len].ljust(cnn_len, "N")
        kmers = get_kmers(seq, K, kmertype)
        
        X_oh.append(weighted_one_hot_kmers(kmers, kmer_to_index))
        X_d2v.append(dna2vec_sequence_matrix(seq, K, kmer_to_index_for_dna2vec))
        y.append(1 if sid in pos else 0)
        ids.append(sid)

    return (
        np.stack(X_oh),
        np.stack(X_d2v),
        np.array(y),
        np.array(ids)
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    valid_kmers = generate_valid_kmers(K)
    kmer_to_index = {k:i for i,k in enumerate(valid_kmers)}
    kmer_to_index_for_dna2vec = build_kmer_dict(K)

    for species in SPECIES:
        print(f"\nProcessing {species}")

        # compute cnn_len globally per species
        lengths = []
        for fold in FOLDS:
            base = f"Splits/{species}/fold{fold}"
            for f in ["train_pos.txt","train_neg.txt","test_pos.txt","test_neg.txt"]:
                seqs = read_fasta_txt(f"{base}/{f}")
                lengths.extend(len(s) for s in seqs.values())

        cnn_len = max(lengths)
        print(f"  CNN length = {cnn_len}")

        for fold in FOLDS:
            print(f"  Fold {fold}")
            base = f"Splits/{species}/fold{fold}"

            Xtr_oh, Xtr_d2v, ytr, idtr = encode_split(
                f"{base}/train_pos.txt",
                f"{base}/train_neg.txt",
                kmer_to_index,
                kmer_to_index_for_dna2vec,
                cnn_len
            )

            Xte_oh, Xte_d2v, yte, idte = encode_split(
                f"{base}/test_pos.txt",
                f"{base}/test_neg.txt",
                kmer_to_index,
                kmer_to_index_for_dna2vec,
                cnn_len
            )

            np.savez_compressed(
                f"{OUTDIR}/{species}_{kmertype}_fold{fold}_train.npz",
                X_oh=Xtr_oh,
                X_d2v=Xtr_d2v,
                y=ytr,
                ids=idtr
            )

            np.savez_compressed(
                f"{OUTDIR}/{species}_{kmertype}_fold{fold}_test.npz",
                X_oh=Xte_oh,
                X_d2v=Xte_d2v,
                y=yte,
                ids=idte
            )
            print("Weighted one-hot shape:", Xtr_oh.shape)   # should be (num_samples, num_kmers, cnn_len)
            print("dna2vec shape:", Xtr_d2v.shape)          # should be (num_samples, cnn_len, embedding_dim)
            print("y shape:", ytr.shape)
            print("First sequence weighted one-hot sum:", Xtr_oh[0].sum())
            print("First sequence dna2vec first vector:", Xtr_d2v[0])  # check first 5 dims

            print("Saved features ✔")