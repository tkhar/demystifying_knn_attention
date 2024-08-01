import torch
import numpy as np
from naive_attention.naive_attention import calculate_attention_batched
from forward_pass.forward_pass import attn_forward_batched
import time

def run_experiment(Q, K, V):

    # Call the naive_attention function.
    start_time = time.time()
    naive_output = calculate_attention_batched(Q, K, V)
    print("Naive time = ", time.time() - start_time)
    # print(naive_output)

    # Call the attn_forward function.
    start_time = time.time()
    attn_output = attn_forward_batched(Q, K, V)
    print("Attn time = ", time.time() - start_time)
    # print(attn_output)

    return torch.mean(torch.abs(naive_output - attn_output))

torch.manual_seed(0)
b = 2
h = 6
n, d = 64, 32
B = 2

# Generate b x h n x d random matrices with entries in [-B,B].
# These are real numbers.
Q = torch.rand(b, h, n, d, requires_grad=False) * 2 * B - B
K = torch.rand(b, h, n, d, requires_grad=False) * 2 * B - B
V = torch.rand(b, h, n, d, requires_grad=False) * 2 * B - B

# Cast to float32
Q = Q.float()
K = K.float()
V = V.float()

num_iterations = 1

estimate = 0
for i in range(num_iterations):
    estimate += run_experiment(Q, K, V)

print("Mean error = ", estimate / num_iterations)