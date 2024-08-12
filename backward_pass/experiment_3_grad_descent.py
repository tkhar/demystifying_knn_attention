# Experiment 2: Gradient on a non-convex loss function using our method vs the naive method.
# Loss function: exp(-O * sin(O))

import torch
from torch import nn
from naive_attention import naive_attention
from naive_backprop import naive_backprop
from backward_pass.backward_pass import fast_grad_V, fast_grad_Q, fast_grad_K
import matplotlib.pyplot as plt
import time
import psutil
import os

def experiment():

    # Embedding dimension
    d = 64

    # Input sequence length
    N = 1000

    # Q,K,V matrices - input to self-attention
    Q = torch.randn(N, d, requires_grad=True)
    K = torch.randn(N, d, requires_grad=True)
    V = torch.randn(N, d, requires_grad=True)

    Q_copy_fast = Q.clone().detach().requires_grad_(True)
    K_copy_fast = K.clone().detach().requires_grad_(True)
    V_copy_fast = V.clone().detach().requires_grad_(True)

    # We'll run gradient descent on the loss function.
    # We'll compare the descent with the gradients calculated using our method, vs
    # the gradients calculated using the naive method.
    iteration = 0
    num_iterations = 10
    losses_fast = []
    losses = []
    dv_average_time = 0
    dQ_average_time = 0
    autograd_average_time = 0

    while iteration != num_iterations:
        iteration += 1
        print(f"Iteration: {iteration}")

        # Calculate naive attention
        O_fast = naive_attention.calculate_attention(Q_copy_fast, K_copy_fast, V_copy_fast) # N x d
        O_fast.retain_grad()

        O = naive_attention.calculate_attention(Q, K, V) # N x d
        O.retain_grad()
        
        # Calculate the loss. Our loss is a convex function.
        loss_fast = torch.mean(torch.exp(-O_fast * torch.sin(O_fast)))
        loss = torch.mean(torch.exp(-O * torch.sin(O)))

        losses_fast.append(loss_fast.item())
        losses.append(loss.item())

        # Calculate the gradients of the loss 

        ## -- BACKWARD PASS -- ##
        start_time = time.time()
        loss.backward(retain_graph=True)
        autograd_average_time += time.time() - start_time

        loss_fast.backward(retain_graph=True)

        ## --- FAST --- ##

        # Approximate the gradient with respect to V:
        time_start = time.time()

        dV_fast = fast_grad_V(Q_copy_fast, K_copy_fast, V_copy_fast, O_fast.grad, epsilon=0.2)
        dv_average_time += time.time() - time_start

        # Print the mean absolute error:
        # print(f"dV: Mean absolute error: {torch.mean(torch.abs(dV_fast - V.grad)).item()}")

        # Approximate the gradient with respect to Q.
        # First, clone the input matrices.
        Q_copy = Q.clone().detach().requires_grad_(False)
        K_copy = K.clone().detach().requires_grad_(False)
        V_copy = V.clone().detach().requires_grad_(False)

        time_start = time.time()
        dQ_fast = fast_grad_Q(Q_copy, K_copy, V_copy, O.grad, epsilon=0.1, delta=0.5)
        dQ_average_time += time.time() - time_start

        # Print the mean absolute error:
        # print(f"dQ: Mean absolute error: {torch.mean(torch.abs(dQ_fast - Q.grad)).item()}")

        # Update the Q, V matrices
        with torch.no_grad():
            Q -= 0.1 * Q.grad
            V -= 0.1 * V.grad
            Q.grad.zero_()
            V.grad.zero_()

            Q_copy_fast -= 0.1 * dQ_fast
            V_copy_fast -= 0.1 * dV_fast

    print(f"Average time for dV: {dv_average_time / num_iterations}")
    print(f"Average time for dQ: {dQ_average_time / num_iterations}")
    print(f"Average time for autograd: {autograd_average_time / num_iterations}")

    # Plot the loss and the loss from the fast gradients
    plt.plot(losses_fast)
    plt.plot(losses)
    plt.legend(['Fast gradients', 'Naive gradients'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss non-convex: exp(-O * sin(O))')
    plt.show()

experiment()