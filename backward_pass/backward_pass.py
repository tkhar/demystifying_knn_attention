import torch
import math
from random_walk_simulation.random_walk import approximate_product, simulate_random_walks
from softmax_expectation.softmax_expectation import softmax_expectation_estimation, softmax_expectation_calculation_with_pre_samples, take_multiple_samples
from softmax_expectation.topk import topk
import numpy as np

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
def fast_grad_V(Q, K, V, dO,epsilon=0.01):
    dV = torch.zeros_like(V)

    # For each column vector of dO, calculate P^T dO:
    for j in range(Q.shape[1]):
        estimated_dV_j = approximate_product(Q, K, dO[:,j], num_samples=math.ceil(Q.shape[1] * math.log(Q.shape[0]) / epsilon))
        dV[:,j] = estimated_dV_j

    return dV

def fast_grad_Q(Q,K,V, dO, epsilon=0.2, delta=0.1):
    dQ = torch.zeros_like(Q)

    n, d = Q.shape
    scores, S = topk(Q, K, int(math.sqrt(n)))
    scores = torch.tensor(scores, dtype=torch.float64)

    for i in range(n):
        for j in range(d):
            def F_1_func(Q, K, V, dO, index, i, j):
                return K[index,j] * (dO[i,:] @ V[index,:])
            
            inputs_1 = (V, dO, i, j)

            E_1 = softmax_expectation_estimation(Q, K, i, F_1_func, inputs_1, S[i,:].tolist(), scores[i,:], epsilon=epsilon, delta=delta)

            def F_2_func(Q, K, V, dO, index, i, j):
                return K[index, j]
            
            inputs_2 = (None, None, None, j)

            E_2 = softmax_expectation_estimation(Q, K, i, F_2_func, inputs_2, S[i,:].tolist(), scores[i,:], epsilon=epsilon, delta=delta)

            def F_3_func(Q, K, V, dO, index, i, j=None):
                return dO[i,:] @ V[index,:]

            inputs_3 = (V, dO, i, None)

            E_3 = softmax_expectation_estimation(Q, K, i, F_3_func, inputs_3, S[i, :].tolist(), scores[i, :], epsilon=epsilon, delta=delta)

            dQ[i,j] = torch.tensor(E_1.item() - (E_2.item() * E_3.item()), dtype=torch.float64)


    return dQ


def dP(dO, V, i, j):
    return dO[i,:] @ V[j,:]

def P(Q, K, i,j):
    distribution = torch.softmax(Q[i,:] @ K.T, dim=0)
    return distribution[j]

def dPP(Q,K,V,dO,i):
    n,d = Q.shape
    ip1 = torch.tensor([dP(dO,V,i,k) for k in range(n)])
    ip2 = torch.tensor([P(Q,K,i,k) for k in range(n)])
    ip = ip1 @ ip2
    
    return ip

# TODO: Can we re-use samples?
def fast_grad_Q_faster(Q,K,V, dO, epsilon=0.2, delta=0.1):
    dQ = torch.zeros_like(Q)

    n, d = Q.shape
    scores, S = topk(Q, K, 4 * int(math.sqrt(n)))
    scores = torch.tensor(scores, dtype=torch.float64)

    print("Topk done.")

    for i in range(n):

        if i % 10 == 0:
            print(f"Processing row {i} out of {n}.")

        idx_sample = take_multiple_samples(Q, K, i, S[i,:].tolist(), scores[i,:], int(1/epsilon), int(np.log(1/delta)))

        def F_3_func(Q, K, V, dO, index, i, j=None):
                return dO[i,:] @ torch.t(V[index,:])

        inputs_3 = (V, dO, i, None)

        E_3 = softmax_expectation_estimation(Q, K, i, F_3_func, inputs_3, S[i, :].tolist(), scores[i, :], epsilon=epsilon, delta=delta)

        for j in range(d):
            def F_1_func(Q, K, V, dO, index, i, j):
                return K[index,j] * (dO[i,:] @ torch.t(V[index,:]))
            
            inputs_1 = (V, dO, i, j)

            E_1 = softmax_expectation_calculation_with_pre_samples(Q, K, F_1_func, inputs_1, idx_sample)

            def F_2_func(Q, K, V, dO, index, i, j):
                return K[index, j]
            
            inputs_2 = (None, None, None, j)

            E_2 = softmax_expectation_calculation_with_pre_samples(Q, K, F_2_func, inputs_2, idx_sample)

            dQ[i,j] = torch.tensor(E_1.item() - (E_2.item() * E_3.item()), dtype=torch.float64)

    return dQ

def fast_grad_K(Q,K,V, dO, epsilon=0.05, delta=0.1):
    dK = torch.zeros_like(K)

    n, d = Q.shape

    S = topk(Q, K, int(math.sqrt(n)))

    
