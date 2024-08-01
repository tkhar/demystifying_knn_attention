import torch
from softmax_expectation.softmax_expectation import softmax_expectation_estimation_faster
from softmax_expectation.topk import topk
import numpy as np

# This function calculates the attention mechanism in the forward pass.
# It uses the softmax expectation approximation method.
def attn_forward(Q, K, V, epsilon=0.01, delta=0.01):

    # Calculate the number of rows in Q.
    n = Q.shape[0]

    # Calculate the number of columns in V.
    d = V.shape[1]

    # Calculate the top sqrt(n) indices of Q[i] @ K[j]^T.
    k = int(n ** 0.5)
    scores, S = topk(Q, K, k)

    S = torch.tensor(S, dtype=torch.int64)
    scores = torch.tensor(scores, dtype=torch.float32)

    # Initialize the output tensor.
    output = torch.zeros_like(V, dtype=torch.float32)

    MM = 20

    # For all rows in Q...
    for i in range(n):
        # For all columns in V...
        denom = torch.sum(torch.exp(scores[i, :]-MM))
        for j in range(d):
            # Approximate the expected value of the value vector.
            
            def f(Q, K, V, dO, index, i, j=None):
                return V[index, j]
            
            inputs = (V, None, None, j)

            output[i, j] = softmax_expectation_estimation_faster(
                Q, K, i, \
                f, \
                inputs, \
                S[i,:].tolist(), \
                scores[i, :].tolist(), \
                denom=denom, \
                epsilon=epsilon, delta=delta).item()

    return output

# This function calculates the attention mechanism in the forward pass.
# Inputs:
# - Q: A tensor of shape (b,h,n,d) containing the query vectors.
# - K: A tensor of shape (b,h,n,d) containing the key vectors.
# - V: A tensor of shape (b,h,n,d) containing the value vectors.
#
# Note that b is the batch size, h is the number of heads, 
# n is the sequence length, and d is the dimension of the vectors.
#
# Outputs:
# - A tensor of shape (b,h,n,d) containing the output vectors.
def attn_forward_batched(Q, K, V, epsilon=0.01, delta=0.01):
        
    B,H,N,D = Q.shape

    k = 2 # int(N ** 0.5)

    # For each batch and head, 
    # Calculate the top sqrt(n) indices of Q[i] @ K[j]^T for all i.
    # The result should be stored in a tensor of shape (B,H,N,k).
    # TODO: Try vectorize this as well.
    S = torch.zeros(B,H,N,k, dtype=torch.int64)
    scores = torch.zeros(B,H,N,k, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            ss, si = topk(Q[b,h], K[b,h], k)
            S[b,h] = torch.tensor(si)
            scores[b,h] = torch.tensor(ss)

    ### -- VECTORIZE THE CODE SO YOU LIVE TO SEE THE RESULTS -- ###
    # 
    # Warning: This code is not easy to understand.
    # Warning 2: This code is not easy to debug.

    # Initialize the output tensor.
    # Compute the denominator for all b, h, i at once
    denom = torch.sum(torch.exp(scores - 20), dim=-1, keepdim=True)

    scores = scores.unsqueeze(3) # B x H x N x 1 x k
    scores = torch.exp(scores - 20) # B x H x N x 1 x k
    scores_3d = scores.view(-1, 1, k) # BH x N x 1 x k

    # Good luck figuring out what this does... 
    BB = torch.arange(V.size(0)).view(-1, 1, 1, 1).expand(-1, V.size(1), V.size(2), -1)
    HH = torch.arange(V.size(1)).view(1, -1, 1, 1).expand(-1, V.size(1), V.size(2), -1)

    V_selected = V[BB, HH, S] # B x H x N x k x D
    V_selected_3d = V_selected.view(-1, k, D)

    numerator = torch.bmm(scores_3d, V_selected_3d)
    numerator = numerator.view(B, H, N, D)

    return numerator / denom


    # -- UNVECTORIZED CODE -- Go get a coffee while you wait for the results -- #
    #
    # for b in range(B):
    #     for h in range(H):
    #         for i in range(N):
    #             denom = torch.sum(torch.exp(torch.tensor(scores[b,h,i,:]) - 20))
    #             for j in range(D):

    #                 numerator = torch.inner(np.exp(torch.tensor(scores[b,h,i,:]) - 20),\
    #                         torch.tensor(V[b,h,S[b,h,i,:],j]))
    #                 output[b,h,i,j] = numerator / denom

    #                 # def f(Q, K, V, dO, index, i, j=None):
    #                 #     return V[index, j]
                    
    #                 # inputs = (V[b,h,:,:], None, None, j)

    #                 # output[b,h,i,j] = softmax_expectation_estimation_faster(
    #                 #                                                     Q[b,h,:,:].view(N,D), \
    #                 #                                                     K[b,h,:,:].view(N,D), \
    #                 #                                                     i, \
    #                 #                                                     f,
    #                 #                                                     inputs,
    #                 #                                                     S[b,h,i,:].tolist(), \
    #                 #                                                     scores[b,h,i,:].tolist(), \
    #                 #                                                     denom=denom, \
    #                 #                                                     epsilon=epsilon, \
    #                 #                                                     delta=delta).item()
    # 
    #                 output[b,h,i,j] = numerator / denom
                                                                         
                