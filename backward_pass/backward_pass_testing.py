import torch
from torch import nn
from naive_attention import naive_attention
from naive_backprop import naive_backprop
from backward_pass.backward_pass import fast_grad_V, fast_grad_Q, fast_grad_K

# This file will test the correctness of self-attention backprop implementation.
# Suppose our loss is the mean squared error loss.
loss = nn.CrossEntropyLoss()

# Vocab size. This is the number of classes in the classification task.
vocab_size = 10

# Embedding dimension
d = 10

# Input sequence length
N = 1000

mu = 20
std = 10

# Q,K,V matrices - input to self-attention
Q = torch.randn(N, d, requires_grad=True) * std + mu
Q.retain_grad()
K = torch.randn(N, d, requires_grad=True) * std + mu
K.retain_grad()
V = torch.randn(N, d, requires_grad=True) * std + mu
V.retain_grad()

# Calculate naive attention
O = naive_attention.calculate_attention(Q, K, V) # N x d
O.retain_grad()

# Create a linear layer
W = torch.tensor(torch.randn(d, vocab_size), requires_grad=True) * std + mu
W.retain_grad()

L = O @ W # N x vocab_size
L.retain_grad()

# Calculate the loss. This is the average cross entropy loss for all N positions in the 
# input sequence.
T = torch.randint(0, vocab_size, (N,)) # N x 1
loss = loss(L, T)

print(f"Loss = {loss.item()}")

# Calculate the gradients of the loss w.r.t. W and O.
loss.backward()

# Sanity check: dO = dL * W.T
# if torch.max(torch.abs(O.grad - L.grad.mm(W.weight))) > 1e-3:
#     print("Gradients of O and L are not consistent.")

# Calculate the gradient with respect to V:
# dV = naive_backprop.grad_V(Q, K, V, O.grad)
# if torch.max(torch.abs(dV - V.grad)) > 0.01:
#     print("Gradients of V are not correct.")
#     print("Error is: ", torch.max(torch.abs(dV - V.grad)))
# else:
#     print("Gradients are correct. Testing fast gradients.")

# Approximate the gradient with respect to V:
dV_fast = fast_grad_V(Q, K, V, O.grad)

# Print the mean absolute error.
print("dV: Mean absolute error:", torch.mean(torch.abs(V.grad - dV_fast)).item())

# Calculate the gradient with respect to Q (naively)
# dQ = naive_backprop.grad_Q(Q, K, V, O.grad)

# Calcuate the gradient with respect to Q (true)
dQ_true = Q.grad
# if torch.max(torch.abs(dQ - dQ_true)) > 0.01:
#     print("Gradients of Q are not correct.")
#     print("Error is: ", torch.max(torch.abs(dQ - dQ_true)))
# else:
#     print("Gradients are correct. Testing fast gradients.")

# Approximate the gradient with respect to Q.
# First, clone the input matrices.
Q_copy = Q.clone().detach().requires_grad_(False)
K_copy = K.clone().detach().requires_grad_(False)
V_copy = V.clone().detach().requires_grad_(False)
dQ_fast = fast_grad_Q(Q_copy, K_copy, V_copy, O.grad, epsilon=1,delta=0.3)

# Print the mean absolute error.
print("dQ: Mean absolute error:", torch.mean(torch.abs(dQ_true - dQ_fast)).item())

print(dQ_true)
print(V.grad)

# Print the mean absolute error with the manually implemented gradient.
# print("dQ: Mean absolute error:", torch.mean(torch.abs(dQ - dQ_fast)).item())

# # Calculate the gradient with respect to K (naively)
# dK = naive_backprop.grad_K(Q, K, V, O.grad)

# # Calcuate the gradient with respect to K (true)
# dK_true = K.grad
# if torch.max(torch.abs(dK - dK_true)) > 0.01:
#     print("Gradients of K are not correct.")
#     print("Error is: ", torch.max(torch.abs(dK - dK_true)))
# else:
#     print("Gradients are correct. Testing fast gradients.")

