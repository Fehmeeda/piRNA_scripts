def build_graph(kmers, kmer_to_index):
    num_nodes = len(kmers)
    node_features = one_hot_encode_kmers(kmers, kmer_to_index)
    
    # Adjacency matrix (connect consecutive nodes)
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes - 1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1  # optional: undirected
    
    return adj, node_features
