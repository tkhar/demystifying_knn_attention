# Experiment 2: Gradient on a non-convex loss function using our method vs the naive method.
# Loss function: 0.6 * (O ** 4) + 2 * (O ** 3) + 0.5 * (O ** 2) + O + 10

import torch
from torch import nn
from naive_attention import naive_attention
from naive_backprop import naive_backprop
from backward_pass.backward_pass import fast_grad_V, fast_grad_Q, fast_grad_K, fast_grad_Q_faster
import matplotlib.pyplot as plt

# Embedding dimension
d = 5

# Input sequence length
N = 15

# Q,K,V matrices - input to self-attention
Q = torch.randn(N, d, requires_grad=True, dtype=torch.float64)
K = torch.randn(N, d, requires_grad=True, dtype=torch.float64) 
V = torch.randn(N, d, requires_grad=True, dtype=torch.float64)

Q_copy_fast = Q.clone().detach().requires_grad_(True)
K_copy_fast = K.clone().detach().requires_grad_(True)
V_copy_fast = V.clone().detach().requires_grad_(True)

# We'll run gradient descent on the loss function.
# We'll compare the descent with the gradients calculated using our method, vs
# the gradients calculated using the naive method.
iteration = 0
num_iterations = 50
losses_fast = []
losses = []
while iteration != num_iterations:
    iteration += 1
    print(f"Iteration: {iteration}")

    # Calculate naive attention
    O_fast = naive_attention.calculate_attention(Q_copy_fast, K_copy_fast, V_copy_fast) # N x d
    O_fast.retain_grad()

    O = naive_attention.calculate_attention(Q, K, V) # N x d
    O.retain_grad()
    
    # Calculate the loss. Our loss is a convex function.
    loss_fast = torch.sum(0.6 * (O_fast ** 4) + 2 * (O_fast ** 3) + 0.5 * (O_fast ** 2) + O_fast +10)
    loss = torch.sum(0.6 * (O ** 4) + 2 * (O ** 3) + 0.5 * (O ** 2) + O + 10)

    losses_fast.append(loss_fast.item())
    losses.append(loss.item())

    # Calculate the gradients of the loss 
    loss.backward(retain_graph=True)
    loss_fast.backward(retain_graph=True)

    # Approximate the gradient with respect to V:
    dV_fast = fast_grad_V(Q_copy_fast, K_copy_fast, V_copy_fast, O_fast.grad, epsilon=0.01)

    # Print the mean absolute error:
    # print(f"dV: Mean absolute error: {torch.mean(torch.abs(dV_fast - V.grad)).item()}")

    # Approximate the gradient with respect to Q.
    # First, clone the input matrices.
    Q_copy = Q.clone().detach().requires_grad_(False)
    K_copy = K.clone().detach().requires_grad_(False)
    V_copy = V.clone().detach().requires_grad_(False)
    dQ_fast = fast_grad_Q(Q_copy, K_copy, V_copy, O.grad, epsilon=0.1, delta=0.5)

    # Print the mean absolute error:
    # print(f"dQ: Mean absolute error: {torch.mean(torch.abs(dQ_fast - Q.grad)).item()}")

    # Update the Q, V matrices
    with torch.no_grad():
        Q -= 0.05 * Q.grad
        V -= 0.05 * V.grad
        Q.grad.zero_()
        V.grad.zero_()

        Q_copy_fast -= 0.05 * dQ_fast
        V_copy_fast -= 0.05 * dV_fast

# Plot the loss and the loss from the fast gradients
plt.plot(losses_fast)
plt.plot(losses)
plt.legend(['Fast gradients', 'Naive gradients'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss non-convex: 0.6 * (O ** 4) + 2 * (O ** 3) + 0.5 * (O ** 2) + O + 10')
plt.show()