import torch
import time
import numpy as np
from clustering.clustering_topk import clustering_topk, clustering_topk_preprocessing_flatl2, clustering_topk_preprocessing_ivff_index
from sampling_attention import sampling_attention
from naive_attention import naive_attention

n, d = 5000, 30

# Generate 50x50 random matrices with elements in range [-10, 10]
Q = torch.randint(-10, 11, (n, d)).float()
K = torch.randint(-10, 11, (n, d)).float()
V = torch.randint(-10, 11, (n, d)).float()

k = l = 200

# Call the clustering_topk_preprocessing function.
# Measure the time it takes to run the function.
start_time = time.time()
index = clustering_topk_preprocessing_ivff_index(K, 10.0, voronoi_cells=100, nprobe=10, verbose=False)

# Get the top-k attention scores and indices.
topk_scores, topk_indices = clustering_topk(Q, K, k, B=10.0, index=index)

print("Time taken for clustering_topk_preprocessing:", time.time() - start_time)

# Call the sampling_attention function.
attention_output = \
    sampling_attention.sampling_attention_clustering_v2(Q, K, V, \
                                                        k, l, \
                                                        topk_scores, \
                                                        topk_indices, \
                                                        B=10.0)
end_time = time.time()
print("Time taken for approximate attention:", end_time - start_time)

# Compare with the exact attention output
# Calculate the time taken to run the calculate_attention function.
start_time = time.time()
exact_attention_output = naive_attention.calculate_attention(Q, K, V, B=10.0)
end_time = time.time()
print("Time taken for exact attention:", end_time - start_time)

# # Print the mean absolute error
# print("Mean error: ", torch.mean(torch.abs(attention_output - exact_attention_output)))