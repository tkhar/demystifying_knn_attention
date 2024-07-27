import torch
import math
from random_walk_simulation.random_walk import approximate_product, simulate_random_walks

# Fast self-attention backpropagation.
# Suppose we have some loss function f.
# Input:
#   - Q: Query matrix of shape: N x d
#   - K: Key matrix of shape: N x d
#   - V: Value matrix of shape: N x d
#   - dO: Derivative of f with respect to the output of the self-attention layer of shape: N x d
# 
# Output:
#   - dV: Approximation of derivative of f with respect to V of shape: N x d
#
# Description:
#  - This algorithm works by using Markov Chain Monte Carlo Simulations.
# 
# Complexity:
#   - O(N * d * log(N) / e^2) where e is the error of the approximation.
#
# Improvements:
#   - Sample from the softmax distribution in O(sqrt(N)) time using the Gumbel-Max trick.
#
def fast_grad_V(Q, K, V, dO,e=1e-1):
    dV = torch.zeros_like(V)

    # For each column vector of dO, calculate P^T dO:
    for j in range(Q.shape[1]):
        estimated_dV_j = approximate_product(Q, K, dO[:,j], num_samples=math.ceil(Q.shape[1] * math.log(Q.shape[0]) / e))
        dV[:,j] = estimated_dV_j

    return dV

def fast_grad_Q(Q,K,V, dO, e=1e-1):
    pass

def fast_grad_K(Q,K,V, dO, e=1e-1):
    pass