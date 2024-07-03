import torch
import time
import numpy as np
from topk_lsh import topk_indices_lsh_preprocessing, topk_indices_fast_lsh
from sampling_attention import sampling_attention
from naive_attention import calculate_attention

# Generate 50x50 random matrices with elements in range [-10, 10]
Q = torch.randint(-20, 21, (200, 30)).float()
K = torch.randint(-20, 21, (200, 30)).float()
V = torch.randint(-20, 21, (200, 30)).float()

# Call the sampling_attention function.
# Measure the time it takes to run the function.
start_time = time.time()
lsh_objects = topk_indices_lsh_preprocessing(K, B=10.0)

print("Pre-processing done in time ", time.time() - start_time)

# for lsh_object in lsh_objects:
#     lsh_object.print_buckets()

attention_output = sampling_attention(Q, K, V, 30,30, topk_indices_fast_lsh, B=10.0, lsh_objects=lsh_objects)
end_time = time.time()
print("Time taken for approximate attention:", end_time - start_time)

# Compare with the exact attention output
# Calculate the time taken to run the calculate_attention function.
start_time = time.time()

# First run the preprocessing step to create the LSH objects.
exact_attention_output = calculate_attention(Q, K, V, B=10.0)
end_time = time.time()
print("Time taken for exact attention:", end_time - start_time)

# Print the mean absolute error
print("Mean error: ", torch.mean(torch.abs(attention_output - exact_attention_output)))