import torch
import faiss

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
def topk(Q, K, k):
    n,d = K.shape
    # We will use the Faiss library to perform the clustering.

    # Build a FAISS Flat index:
    index = faiss.IndexFlatIP(d)

    K_normalized = K.clone()

    # Add the normalized key vectors to the index
    index.add(K_normalized.numpy().astype('float32'))

    # Search for each query vector in Q in the index for
    # its k nearest neighbors.
    # This takes O(nk) time.
    _, I = index.search(Q.numpy().astype('float32'), k)

    return I