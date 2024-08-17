import torch
import math
import numpy as np
import faiss
import time

# 
# Find the top k indices of Q[i] @ K[j]^T for j in [n] and all i in [n]
#
# Input:
#   - Q: A matrix of shape n x d -> This is the query matrix.
#   - K: A matrix of shape n x d -> This is the key matrix.
#   - k: The number of indices to return.
#
# Output:
#   - The top k indices of Q[i] @ K[j]^T.
#
def topk(Q, K, k):
	n, d = Q.shape
    # We will use the Faiss library to perform the clustering.

    # Build a FAISS Flat index:
	index = faiss.IndexFlatIP(d)

	K_normalized = K.clone().detach().requires_grad_(False)
	Q_clone = Q.clone().detach().requires_grad_(False)

	# Add the normalized key vectors to the index
	index.add(K_normalized.cpu().float().numpy())

    # Search for each query vector in Q in the index for
    # its k nearest neighbors.
    # This takes O(nk) time.
	scores, I = index.search(Q_clone.cpu().float().numpy(), k)

	return scores, I


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
def attn_forward_batched(Q, K, V):
        
	B,H,N,D = Q.shape

	k = 10 # 4 * int(N ** 0.5)
    
	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"
		
	Q = Q.to(device)
	K = K.to(device)
	V = V.to(device)
        
	with torch.cuda.device(device):
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

		scores /= (D ** 0.5)

		# We'll create a mask for attention to attend only to the left of the current position.
		indices = torch.arange(N)[:, None].expand(N, k)
		indices = indices[None, None, :, :].expand(B, H, N, k)
		mask = (S > indices)
		S[mask] = N
		# Add a row of zeros to V for every batch and head.
		# Compute the denominator for all b, h, i at once
		ones_denom = torch.ones_like(V)
		V = torch.cat((V, torch.zeros(B,H,1,D, dtype=V.dtype).to(device)), dim=2)
		ones_denom = torch.cat((ones_denom, torch.zeros(B,H,1,D, dtype=ones_denom.dtype).to(device)), dim=2)
		S = torch.cat((S, N * torch.ones(B,H,1,k, dtype=S.dtype)), dim=2)

		MM = torch.max(scores)

		scores = scores.unsqueeze(3) # B x H x N x 1 x k
		scores = torch.exp(scores - MM) # B x H x N x 1 x k
		scores_3d = scores.view(-1, 1, k).to(device) # BH x N x 1 x k

		# Good luck figuring out what this does... 
		BB = torch.arange(V.size(0)).view(-1, 1, 1, 1).expand(-1, V.size(1), V.size(2), -1)
		HH = torch.arange(V.size(1)).view(1, -1, 1, 1).expand(-1, V.size(1), V.size(2), -1)

		# Now we want to select k rows from V for each b, h, i according to S[b,h,i]
		V_selected = V[BB,HH,S]
		V_selected = V_selected[:, :, :-1, :] # Remove the row of zeros we added earlier
		V_selected_3d = V_selected.reshape(-1, k, D).to(device)

		numerator = torch.bmm(scores_3d, V_selected_3d)
		numerator = numerator.view(B, H, N, D)

		ones_denom_selected = ones_denom[BB,HH,S]
		ones_denom_selected = ones_denom_selected[:, :, :-1, :] # Remove the row of zeros we added earlier
		ones_denom_selected = ones_denom_selected.reshape(-1, k, D)
		denom = torch.bmm(scores_3d, ones_denom_selected)
		denom = denom.view(B, H, N, D)

		output = numerator / denom

		# Replace NaNs with zeros
		output[torch.isnan(output)] = 1

		return output
