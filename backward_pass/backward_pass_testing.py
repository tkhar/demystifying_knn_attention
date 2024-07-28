import torch
from torch import nn
from naive_attention import naive_attention
from naive_backprop import naive_backprop
from backward_pass.backward_pass import fast_grad_V, fast_grad_Q, fast_grad_K, fast_grad_Q_faster

# This file will test the correctness of self-attention backprop implementation.
# Suppose our loss is the mean squared error loss.
loss = nn.CrossEntropyLoss()

# Vocab size. This is the number of classes in the classification task.
vocab_size = 10

# Embedding dimension
d = 5

# Input sequence length
N = 20

B = 4

# Q,K,V matrices - input to self-attention
# Their elements are drawn from a uniform distribution from -B to B.
# To sample from a uniform distribution from -B to B, we can sample from a uniform distribution from 0 to 2B and subtract B.
Q = torch.rand(N, d, requires_grad=True) * 2 * B - B
Q.retain_grad()
K = torch.rand(N, d, requires_grad=True) * 2 * B - B
K.retain_grad()
V = torch.rand(N, d, requires_grad=True) * 2 * B - B
V.retain_grad()


# Calculate naive attention
O = naive_attention.calculate_attention(Q, K, V) # N x d
O.retain_grad()

# Have the linear layer be a constant matrix: d x vocab_size full of 1s
# W = torch.ones(d, vocab_size, requires_grad=True)
# W.retain_grad()

# L = O @ W # N x vocab_size
# L.retain_grad()

# # Calculate the loss. This is the average cross entropy loss for all N positions in the 
# # # input sequence.
# T = torch.randint(0, vocab_size, (N,)) # N x 1
# loss = loss(L, T)

# Experiment 1) Our loss is the Frobenius norm of the O matrix.
# loss = 10000 * torch.norm(O, p='fro') 

# Experiment 2) Our loss is the sum of the cubes elements of the O matrix.
loss = torch.sum(O ** 3)


print(f"Loss = {loss.item()}")

# Calculate the gradients of the loss 
loss.backward()

# Sanity check: dO = dL * W.T
# if torch.max(torch.abs(O.grad - L.grad.mm(W.weight))) > 1e-3:
#     print("Gradients of O and L are not consistent.")

# Calculate the gradient with respect to V:
# dV = naive_backprop.grad_V(Q, K, V, O.grad)

# If the max relative error between dV and V.grad is greater than 0.01, then the gradients are not correct.
# if torch.max(torch.abs(dV-V.grad) / torch.abs(V.grad)) > 0.01:
#     print("Gradients of V are not correct.")
#     print("Error is: ", torch.max(torch.abs(dV - V.grad)))
# else:
#     print("Gradients of V are correct. Testing fast gradients.")

# Approximate the gradient with respect to V:
dV_fast = fast_grad_V(Q, K, V, O.grad, epsilon=0.01)

# Print the mean relative error. Don't calculate it where dV is 0 because the relative error is not defined.
dV_relative_error = 0
denom_v = 0
for i in range(N):
    for j in range(d):
        if V.grad[i,j] > 0.01:
            denom_v += 1
            dV_relative_error += torch.abs((V.grad[i,j] - dV_fast[i,j]) / V.grad[i,j])

dV_relative_error /= (denom_v)
print(f"dV: Mean relative error: {dV_relative_error.item()*100}%")

# Calculate the gradient with respect to Q (naively)
dQ = naive_backprop.grad_Q(Q, K, V, O.grad)

# Calcuate the gradient with respect to Q (true)
if torch.max(torch.abs(dQ-Q.grad) / torch.abs(Q.grad)) > 0.01:
    print("Gradients of Q are not correct.")
    print("Error is: ", torch.max(torch.abs(dQ - Q.grad)))
else:
    print("Gradients of Q are correct. Testing fast gradients.")

# Approximate the gradient with respect to Q.
# First, clone the input matrices.
Q_copy = Q.clone().detach().requires_grad_(False)
K_copy = K.clone().detach().requires_grad_(False)
V_copy = V.clone().detach().requires_grad_(False)
dQ_fast = fast_grad_Q_faster(Q_copy, K_copy, V_copy, O.grad, epsilon=0.005,delta=0.3)
# dQ_fast = fast_grad_Q(Q_copy, K_copy, V_copy, O.grad, epsilon=0.001, delta=0.3)

for i in range(N):
    for j in range(d):
        if Q.grad[i,j] > 0.01:
            print(f"i: {i}, j: {j}, dQ: {Q.grad[i,j]}, dQ_fast: {dQ_fast[i,j]}")
            print(f"Relative error: {torch.abs((Q.grad[i,j] - dQ_fast[i,j]) / Q.grad[i,j])}")
            # dQ_relative_errors.append(torch.abs((Q.grad[i,j] - dQ_fast[i,j]) / Q.grad[i,j]))

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

