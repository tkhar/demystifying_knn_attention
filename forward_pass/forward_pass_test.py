import torch
import numpy as np
from naive_attention.naive_attention import calculate_attention
from forward_pass.forward_pass import attn_forward


def run_experiment(Q, K, V, n, d):

    # Call the naive_attention function.
    naive_output = calculate_attention(Q, K, V)
    # print(naive_output)

    # Call the attn_forward function.
    attn_output = attn_forward(Q, K, V)
    # print(attn_output)

    return torch.max(torch.abs(naive_output - attn_output))

torch.manual_seed(0)
n, d = 100, 50
B = 100

# Generate n x d random matrices with entries in [-B,B].
# These are real numbers.
Q = torch.rand(n, d) * 2 * B - B
K = torch.rand(n, d) * 2 * B - B
V = torch.rand(n, d) * 2 * B - B

# Cast to float32
Q = Q.float()
K = K.float()
V = V.float()

num_iterations = 1

estimate = 0
for i in range(num_iterations):
    estimate += run_experiment(Q, K, V, n, d)

print("Max error = ", estimate / num_iterations)