# This file contains functions to calculate the expectation of a function with respect to the softmax distribution.
# We do so by sampling from the softmax distribution using the Gumbel-Max trick.
# We only need O(sqrt{n}) time to sample from the softmax distribution.

import torch
import numpy as np

#
# Sample from the softmax distribution over [n].
# Input:
#   - Q: A matrix of shape n x d -> This is the query matrix.
#   - K: A matrix of shape n x d -> This is the key matrix.
#        Note that our stochastic matrix P is given by P = softmax(Q @ K^T).
#   - i: The index of the row of Q to sample from.
#        Note that this means we are sampling from softmax(Q[i] @ K^T).
#   - S_i: The top sqrt(n) indices j of Q[i] @ K[j]^T. To retrieve these indices, use the topk function.
#
# Output:
#   - The index of the sampled element.
#
def softmax_sample(Q, K, i, S_i):
    k = len(S_i)
    n = Q.shape[0]

    # Sample k random numbers from the Gumbel distribution.
    G = -torch.log(-torch.log(torch.rand(k)))

    # Find the max of Q[i] @ K[j]^T + G_j for j in S_i.
    # Also find the index in S_i that achieves this max.
    M = torch.max(Q[i] @ K[S_i,:].T + G)

    # Find the min Q[i] @ K[j]^T for j in S_i.
    s_min = torch.min(Q[i] @ K[S_i,:].T)

    # The threshold to overcome for the Gumbel Max Trick is:
    B = M - s_min

    # Sample m according to a binomial distribution with parameters n-k, 1-exp(-exp(-B))
    m = torch.distributions.Binomial(n-k, 1-torch.exp(-torch.exp(-B))).sample().to(torch.int32)

    # Sample m points from the [n] - S_i set.
    S = []
    while len(S) < m:
        index = np.random.randint(n)
        if index not in S_i:
            S.append(index)

    # Sample a Gumbel random variable for each element in S, conditioned on it being greater than B.
    # To do this, sample a uniform random variable in [exp(-exp(-B)), 1] and let G' = -log(-log(U)).
    U = torch.rand(m) * (1 - torch.exp(-torch.exp(-B))) + torch.exp(-torch.exp(-B))
    G_prime = -torch.log(-torch.log(U))

    # Find the arg max of Q[i] @ K[j]^T + G' for j in S cup S_i.
    # This is the index of the sampled element.
    maximum = -1e9
    index = -1
    k = 0
    for j in S:
        if Q[i] @ K[j,:].T + G_prime[k] > maximum:
            maximum = Q[i] @ K[j,:].T + G_prime[k]
            index = j
        k += 1

    k = 0
    for j in S_i:
        if Q[i] @ K[j,:].T + G[k] > maximum:
            maximum = Q[i] @ K[j,:].T + G[k]
            index = j
        k += 1

    return index

# 
# Calculate the expectation of f with respect to the softmax distribution.
# 
# Input:
# - Q: A matrix of shape n x d -> This is the query matrix.
# - K: A matrix of shape n x d -> This is the key matrix.
# - i: The index of the row of Q to sample from.
# - f: A function that takes an index j and returns a scalar.
# - S_i: The top sqrt(n) indices j of Q[i] @ K[j]^T.
# 
# Output:
# - The expectation of f with respect to the softmax distribution.   
def softmax_expectation(Q, K, i, f, S_i,epsilon=0.1,delta=0.1):
    n = Q.shape[0]
    k = len(S_i)

    # We will use a median-of-means approach to estimate the expectation.
    # In this approach, we repeat the following process O(log(1/delta)) times:
    #    Sample O(1/epsilon^2) elements from the softmax distribution.
    #    Take the mean of these samples. -> M_k
    # Take the median of all the M_k's.
    # This gives us an estimate of the expectation.

    expectations = []
    for _ in range(int(np.log(1/delta))):
        samples = []
        for _ in range(int(1/epsilon**2)):
            index = softmax_sample(Q, K, i, S_i)
            samples.append(f(index))
        expectations.append(np.mean(samples))

    return np.median(expectations)