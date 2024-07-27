import torch
from softmax_expectation.softmax_expectation import softmax_sample, softmax_expectation_estimation
from softmax_expectation.topk import topk

# This function calculates the attention mechanism in the forward pass.
# It uses the softmax expectation approximation method.
def attn_forward(Q, K, V, epsilon=0.01, delta=0.01):

    # Calculate the number of rows in Q.
    n = Q.shape[0]

    # Calculate the number of columns in V.
    d = V.shape[1]

    # Calculate the top sqrt(n) indices of Q[i] @ K[j]^T.
    S = topk(Q, K, int(n ** 0.5))

    # Initialize the output tensor.
    output = torch.zeros_like(V)

    # For all rows in Q...
    for i in range(n):
        # For all columns in V...
        for j in range(d):
            # Approximate the expected value of the value vector.
            # TODO: We can re-use samples over all columns to speed up things.
            output[i, j] = softmax_expectation_estimation(Q, K, i, V[:,j], S[i,:].tolist(), epsilon=epsilon, delta=delta).item()

    return output