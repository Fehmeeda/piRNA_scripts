# Length of all samples including positive and negatives samples from 3 species
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# ============================
# Read FASTA
# ============================
def read_fasta_txt(filename):
    sequences = {}
    with open(filename, 'r') as f:
        seq_id = ''
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq_id:
                    sequences[seq_id] = seq
                seq_id = line[1:]
                seq = ''
            else:
                seq += line.upper()
        if seq_id:
            sequences[seq_id] = seq
    return sequences


# ============================
# Main
# ============================
if __name__ == "__main__":

    # Load Human sequences
    pos_file_Human = 'Human_posi_samples.txt'
    neg_file_Human = 'Human_nega_samples.txt'

    pos_seqs_Human = read_fasta_txt(pos_file_Human)
    neg_seqs_Human = read_fasta_txt(neg_file_Human)

    # Combine for analysis
    all_seqs_Human = {**pos_seqs_Human, **neg_seqs_Human}

    # Compute lengths
    seq_lengths_Human = [len(seq) for seq in all_seqs_Human.values()]
    min_len_Human = min(seq_lengths_Human)
    max_len_Human = max(seq_lengths_Human)

    print(f"Minimum length of Human Sequences: {min_len_Human}")
    print(f"Maximum length of Human Sequences: {max_len_Human}")

    # Count occurrences
    length_counts_Human = Counter(seq_lengths_Human)
    #print(length_counts_Human)

    # Create table
    lengths_Human = list(range(min_len_Human, max_len_Human + 1))
    counts_Human = [length_counts_Human.get(l, 0) for l in lengths_Human]

    length_df_Human = pd.DataFrame({
        "sequence_length": lengths_Human,
        "count": counts_Human
    })

    # Save table
    length_df_Human.to_csv("sequence_length_distribution_Human.csv", index=False)
    print("Saved: sequence_length_distribution_Human.csv")

    # Plot histogram (bar chart)
    plt.figure(figsize=(10, 5))
    plt.bar(lengths_Human, counts_Human, edgecolor="black")
    plt.xticks(lengths_Human)
    plt.xlabel("Sequence Length of Human")
    plt.ylabel("Count")
    plt.title("Histogram of Sequence Lengths of Human")
    plt.tight_layout()
    plt.savefig("sequence_length_histogram_Human.png", dpi=300)
    plt.show()

    print("Saved: sequence_length_histogram_Human.png")

    ###############################
    # MOUSE 
    ##############################

    # Load Mouse sequences
    pos_file_Mouse = 'Mouse_posi_samples.txt'
    neg_file_Mouse = 'Mouse_nega_samples.txt'

    pos_seqs_Mouse = read_fasta_txt(pos_file_Mouse)
    neg_seqs_Mouse = read_fasta_txt(neg_file_Mouse)

    # Combine for analysis
    all_seqs_Mouse = {**pos_seqs_Mouse, **neg_seqs_Mouse}

    # Compute lengths
    seq_lengths_Mouse = [len(seq) for seq in all_seqs_Mouse.values()]
    min_len_Mouse = min(seq_lengths_Mouse)
    max_len_Mouse = max(seq_lengths_Mouse)

    print(f"Minimum length Mouse Sequence: {min_len_Mouse}")
    print(f"Maximum length Mouse Sequence: {max_len_Mouse}")

    # Count occurrences
    length_counts_Mouse = Counter(seq_lengths_Mouse)
    #print(length_counts)

    # Create table
    lengths_Mouse  = list(range(min_len_Mouse , max_len_Mouse  + 1))
    counts_Mouse  = [length_counts_Mouse .get(l, 0) for l in lengths_Mouse ]

    length_df_Mouse  = pd.DataFrame({
        "sequence_length": lengths_Mouse ,
        "count": counts_Mouse 
    })

    # Save table
    length_df_Mouse.to_csv("sequence_length_distribution_Mouse.csv", index=False)
    print("Saved: sequence_length_distribution_Mouse.csv")

    # Plot histogram (bar chart)
    plt.figure(figsize=(10, 5))
    plt.bar(lengths_Mouse, counts_Mouse, edgecolor="black")
    plt.xticks(lengths_Mouse)
    plt.xlabel("Sequence Length of Mouse")
    plt.ylabel("Count")
    plt.title("Histogram of Sequence Lengths of Mouse")
    plt.tight_layout()
    plt.savefig("sequence_length_histogram_Mouse.png", dpi=300)
    plt.show()

    print("Saved: sequence_length_histogram_Mouse.png")

    ###############################
    # DROSOPHILA 
    ##############################

    # Load Drosophila sequences
    pos_file_Drosophila = 'Drosophila_posi_samples.txt'
    neg_file_Drosophila = 'Drosophila_nega_samples.txt'

    pos_seqs_Drosophila = read_fasta_txt(pos_file_Drosophila)
    neg_seqs_Drosophila = read_fasta_txt(neg_file_Drosophila)

    # Combine for analysis
    all_seqs_Drosophila = {**pos_seqs_Drosophila, **neg_seqs_Drosophila}

    # Compute lengths
    seq_lengths_Drosophila = [len(seq) for seq in all_seqs_Drosophila.values()]
    min_len_Drosophila = min(seq_lengths_Drosophila)
    max_len_Drosophila = max(seq_lengths_Drosophila)

    print(f"Minimum length Drosophila Sequence: {min_len_Drosophila}")
    print(f"Maximum length Drosophila Sequence: {max_len_Drosophila}")

    # Count occurrences
    length_counts_Drosophila = Counter(seq_lengths_Drosophila)
    #print(length_counts)

    # Create table
    lengths_Drosophila  = list(range(min_len_Drosophila , max_len_Drosophila  + 1))
    counts_Drosophila  = [length_counts_Drosophila .get(l, 0) for l in lengths_Drosophila ]

    length_df_Drosophila  = pd.DataFrame({
        "sequence_length": lengths_Drosophila ,
        "count": counts_Drosophila 
    })

    # Save table
    length_df_Drosophila.to_csv("sequence_length_distribution_Drosophila.csv", index=False)
    print("Saved: sequence_length_distribution_Drosophila.csv")

    # Plot histogram (bar chart)
    plt.figure(figsize=(10, 5))
    plt.bar(lengths_Drosophila, counts_Drosophila, edgecolor="black")
    plt.xticks(lengths_Drosophila)
    plt.xlabel("Sequence Length of Drosophila")
    plt.ylabel("Count")
    plt.title("Histogram of Sequence Lengths of Drosophila")
    plt.tight_layout()
    plt.savefig("sequence_length_histogram_Drosophila.png", dpi=300)
    plt.show()

    print("Saved: sequence_length_histogram_Drosophila.png")