# ============================
# Function to read FASTA/TXT
# ============================
def read_fasta_txt(path):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seqs.append(line.upper())
    return seqs


# ============================
# Function to calculate GC%
# ============================
def gc_percent(seq):
    if len(seq) == 0:
        return 0
    gc = seq.count("G") + seq.count("C")
    return (gc / len(seq)) * 100


# ============================
# Load your datasets
# ============================
positive_file = "Datasets/Human_posi_samples.txt"    
negative_file = "Datasets/Human_nega_samples.txt"    

pos_seqs = read_fasta_txt(positive_file)
neg_seqs = read_fasta_txt(negative_file)


# ============================
# Compute GC% for all sequences
# ============================
pos_gc_list = [gc_percent(s) for s in pos_seqs]
neg_gc_list = [gc_percent(s) for s in neg_seqs]

# ============================
# Print summary
# ============================
print("=====================================")
print(" GC CONTENT SUMMARY")
print("=====================================")

print(f"Total Positive Sequences: {len(pos_seqs)}")
print(f"Total Negative Sequences: {len(neg_seqs)}\n")

print(f"Average GC% (Positive): {sum(pos_gc_list) / len(pos_gc_list):.2f}")
print(f"Average GC% (Negative): {sum(neg_gc_list) / len(neg_gc_list):.2f}\n")

print(f"Min GC% (Positive): {min(pos_gc_list):.2f}")
print(f"Max GC% (Positive): {max(pos_gc_list):.2f}")

print(f"Min GC% (Negative): {min(neg_gc_list):.2f}")
print(f"Max GC% (Negative): {max(neg_gc_list):.2f}")
