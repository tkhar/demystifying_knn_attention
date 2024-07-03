import numpy as np
import torch
import time

#
# Approximating the expected value of a value vector v with underlying distribution
# p(i) = softmax(q^T k_i) / Z, where Z is the partition function
# 
# Parameters:
#   q is the query vector: 1 x d
#   K is the matrix of key vectors: n x d
#   v is the value vector: 1 x n
#   k is the number of top-k elements to consider
#   l is the number of samples to draw from the remaining elements
#   topk_indices_func is a function that returns the top-k indices of the attention scores
#   B is the maximum value of the q,k,v elements. We use this to avoid numerical instability.
# 
def approximate_softmax_expectation(q, K, v, k, l, topk_indices_func, B, lsh_objects=None):
    n, d = K.size()

    # Find the top-k indices of the attention scores
    scores, indices = topk_indices_func(q, K, k, B, lsh_objects)

    # From the n-k remaining elements, draw l samples.
    random_indices = []
    while len(random_indices) < l:
        index = np.random.randint(n)
        if index not in indices:
            random_indices.append(index)

    # Now we'll evaluate the partition function and the expectation separately.

    approx_partition = 0
    approx_expectation = 0
    for index in random_indices:
        # Calculate the attention score for the remaining elements
        attention_score = torch.exp(torch.dot(q, K[index]) / d - B*B)

        # Add the attention score to the partition function
        approx_partition += attention_score

        # Add the attention score times the value to the expectation
        approx_expectation += attention_score * v[index]

    approx_partition *= ((n-k) / l)
    approx_expectation *= ((n-k) / l)

    approx_partition += scores.sum()
    approx_expectation += torch.sum(scores * v[indices])

    # Return the approximate softmax partition function.
    return approx_expectation / approx_partition


# 
# Approximates the attention output using sampling.
# 
# Parameters:
#   Q is the matrix of query vectors: n x d
#   K is the matrix of key vectors: n x d
#   V is the matrix of value vectors: n x d
#   k is the number of top-k elements to consider
#   l is the number of samples to draw from the remaining elements
#   topk_indices_func is the function to find the top-k indices
#
def sampling_attention(Q, K, V, k, l, topk_indices_func, B, lsh_objects=None):

    output = torch.zeros_like(V)

    # For all rows in Q...
    for i in range(Q.size(0)): # n
        # For all columns in V...
        for j in range(V.size(1)): # d
            # print(i,j)
            # Approximate the expected value of the value vector

            # Time this function
            # import time
            # start_time = time.time()
            output[i, j] = approximate_softmax_expectation(Q[i], K, V[:,j], k, l, topk_indices_func, B, lsh_objects)
            # print("Time taken for expectation: ", time.time() - start_time)

    return output