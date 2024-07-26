import torch
from torch import nn
from naive_attention import naive_attention
from naive_backprop import naive_backprop
from fast_gradients import fast_gradients

# This file will test the correctness of self-attention backprop implementation.
# First, suppose we have a loss function f, which will be the cross-entropy loss.
loss = nn.CrossEntropyLoss()

# Vocab size. This is the number of classes in the classification task.
vocab_size = 10

# Embedding dimension
d = 5

# Input sequence length
N = 10

B = 100

# Q,K,V matrices - input to self-attention
# Elements are drawn uniformly at random from [-B,B] 
Q = torch.randint(-B,B,(N,d)).float() # N x d
K = torch.randint(-B,B,(N,d)).float() # N x d
V = torch.randint(-B,B,(N,d)).float() # N x d

# Calculate naive attention
O = naive_attention.calculate_attention(Q, K, V,retain_grad=True) # N x d
O.retain_grad()

# Create a linear layer
W = nn.Linear(d, vocab_size) # d x V
L = W(O) # N x V
L.retain_grad()

# Calculate the loss. This is the average cross entropy loss for all N positions in the 
# input sequence.
T = torch.randint(0, vocab_size, (N,)) # N x 1
logits = loss(L, T)
print(f"Loss = {logits.item()}")

# Calculate the gradients of the loss w.r.t. W and O.
logits.backward()


# Sanity check: dO = dL * W.T
if torch.max(torch.abs(O.grad - L.grad.mm(W.weight))) > 1e-3:
    print("Gradients of O and L are not consistent.")

# Calculate the gradient with respect to V:
dV = naive_backprop.grad_V(Q, K, V, O.grad)
# print(dV)

# Calculate an approximation to the gradient.
dV_approx = fast_gradients.fast_grad_V(Q,K,V,O.grad, e=1e-2)
# print(dV_approx)

# Find the relative error, ignoring the case where the true gradient is zero.
relative_error = torch.abs(dV - dV_approx) / torch.abs(dV)
relative_error = relative_error[dV != 0]
print(f"Relative error = {torch.mean(relative_error).item()}")

