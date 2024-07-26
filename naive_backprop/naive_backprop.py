import torch


# Naive self-attention backpropagation.
# Suppose we have some loss function f.
# Input:
#   - Q: Query matrix of shape: N x d
#   - K: Key matrix of shape: N x d
#   - V: Value matrix of shape: N x d
#   - dO: Derivative of f with respect to the output of the self-attention layer of shape: N x d
# 
# Output:
#   - dV: Derivative of f with respect to V of shape: N x d
def grad_V(Q, K, V, dO):
    dV = torch.zeros_like(V)

    # Formula for derivative of f with respect to V:
    # dV = P^T * dO
    # P = softmax(QK^T) -> N x N

    # Compute P
    P = torch.mm(Q, K.t())
    P = torch.nn.functional.softmax(P, dim=1)

    # Compute dV
    dV = torch.mm(P.t(), dO)

    return dV