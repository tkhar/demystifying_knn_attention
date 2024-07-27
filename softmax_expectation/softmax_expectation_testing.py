import torch
from softmax_expectation.softmax_expectation import softmax_sample, softmax_expectation_estimation
from softmax_expectation.topk import topk
import matplotlib.pyplot as plt

# Input sequence length
N = 100
d = 10

mu = 2
std = 0.01

Q = torch.randn(N, d) * std + mu
K = torch.randn(N, d) * std + mu

# Sample from the softmax Q[0] @ K[j]^T for j in [N].
# Do this 100 times and form a histogram of the indices.
histogram_true = torch.zeros(N)
distribution = torch.softmax(Q[0] @ K.T, dim=0)
for _ in range(100):
    index = torch.multinomial(distribution, 1).item()
    histogram_true [index] += 1

# Now we do the same thing but with our method.
# But first, our method requires the top sqrt(N) indices of Q[0] @ K[j]^T.

S_i = topk(Q, K, int(N ** 0.5))
histogram_sampled = torch.zeros(N)
for _ in range(100):
    index = softmax_sample(Q, K, 0, S_i[0,:].tolist())
    histogram_sampled[index] += 1

# Draw the two histograms.
# The two histograms should be similar. If they are not, there is a bug in the sampling method.
# plt.bar(range(N), histogram_true)
# plt.bar(range(N), histogram_sampled)
# plt.show()

# Now we will test the expectation method.
# We will calculate the expectation of the function f(j) = j.
# This is the expected value of the sampled index.
# Analytically, we just have to evaluate the inner product of (Q[0] @ K[j]^T)_{j=1}^n * f.

expectation_true = torch.sum(distribution * torch.arange(N))
expectation_sampled = softmax_expectation_estimation(Q, K, 0, torch.arange(N), S_i[0,:].tolist(), epsilon=0.05, delta=0.05)

print(f"True expectation: {expectation_true}")
print(f"Sampled expectation: {expectation_sampled}")
print(f"Error: {torch.abs(expectation_true - expectation_sampled)}")