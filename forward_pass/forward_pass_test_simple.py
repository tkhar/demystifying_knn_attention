import torch
import numpy as np
from naive_attention import naive_attention
from forward_pass.forward_pass import attn_forward


def run_experiment(Q, K, V, n, d):

    # Call the naive_attention function.
    naive_output = naive_attention.calculate_attention(Q, K, V)

    # Call the attn_forward function.
    attn_output = attn_forward(Q, K, V)

    return torch.max(torch.abs(naive_output - attn_output))

torch.manual_seed(0)
n, d = 200, 2
B = 1

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

print("Mean max error = ", estimate / num_iterations)