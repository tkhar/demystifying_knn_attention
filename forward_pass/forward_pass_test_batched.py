import torch
import numpy as np
from naive_attention.naive_attention import calculate_attention_batched
from forward_pass.forward_pass import attn_forward_batched
import time
import tracemalloc
import math
import matplotlib.pyplot as plt

def run_experiment(Q, K, V, k=2):

    # Call the naive_attention function.
    # start_time = time.time()
    # naive_output = calculate_attention_batched(Q, K, V)
    # print("Naive time = ", time.time() - start_time)
    # print(naive_output)

    # Call the attn_forward function.
    start_time = time.time()
    attn_output = attn_forward_batched(Q, K, V, k=k)
    print("Attn time = ", time.time() - start_time)
    # print(attn_output)

    return (0,0) # torch.mean(torch.abs(naive_output - attn_output)), torch.max(torch.abs(naive_output - attn_output))

torch.manual_seed(0)
b = 1
h = 10
n, d = 100000, 32

# Range k.
kk = [int(n ** 0.25)]#, int(n ** 0.25)]#, int(n ** 0.125), int(math.log(n)), 3]
kk = sorted(list(set(kk)))
kk_labels = [r"$n^{1/4}$"]#, r"$n^{1/2}$"] #r"2", r"$\log(n)$", r"$n^{1/8}$", r"$n^{1/4}$", r"$n^{1/2}$"]
if len(kk_labels) != len(kk):
    print(kk)
    print(kk_labels)
    raise ValueError("kk_labels and kk must have the same length.")

# Range B.
BB = [2]
colors = ['b']#, 'r', 'g']

means = []
maxs = []
std_devs_max = []
std_devs_mean = []
for i, B in enumerate(BB):
    means.append([])
    maxs.append([])
    std_devs_max.append([])
    std_devs_mean.append([])
    for k in kk:
        # Generate b x h n x d random matrices with entries in [-B,B].
        # These are real numbers.
        Q = torch.rand(b, h, n, d, requires_grad=False) * 2 * B - B
        K = torch.rand(b, h, n, d, requires_grad=False) * 2 * B - B
        V = torch.rand(b, h, n, d, requires_grad=False) * 2 * B - B

        # Cast to float32
        Q = Q.float()
        K = K.float()
        V = V.float()

        print("B = ", B, "k = ", k)

        num_iterations = 1
        
        results_mean = []
        results_max = []
        for _ in range(num_iterations):
            results = run_experiment(Q, K, V, k=k)
            results_mean.append(results[0])
            results_max.append(results[1])

        results_mean = np.array(results_mean)
        results_max = np.array(results_max)

        means[i].append(np.mean(results_mean))
        maxs[i].append(np.mean(results_max))
        std_devs_mean[i].append(np.std(results_mean))
        std_devs_max[i].append(np.std(results_max))

        print("Mean error = ", means[i][-1], "with std dev = ", std_devs_mean[i][-1])
        print("Max error = ", maxs[i][-1], "with std dev = ", std_devs_max[i][-1])

# Generate two subplots, one for the mean and one for the max.
fig, axs = plt.subplots(2)

# Plot the mean.
for i, B in enumerate(BB):
    axs[0].plot(kk_labels, means[i], label=f"B = {B}", color=colors[i])
    means[i] = np.array(means[i])
    std_devs_mean[i] = np.array(std_devs_mean[i])
    axs[0].fill_between(kk_labels, means[i] - std_devs_mean[i], means[i] + std_devs_mean[i], color=colors[i], alpha=0.2)

# Plot the max.
for i, B in enumerate(BB):
    axs[1].plot(kk_labels, maxs[i], label=f"B = {B}", color=colors[i])
    maxs[i] = np.array(maxs[i])
    axs[1].fill_between(kk_labels, maxs[i] - std_devs_max[i], maxs[i] + std_devs_max[i], color=colors[i], alpha=0.2)

# Set the labels. For $k$ we use the labels in kk_labels.
axs[0].set_title("Mean Approximation Error of kNN Attention")
axs[0].set_ylabel("Mean error")
axs[0].legend()

axs[1].set_title("Max Approximation Error of kNN Attention")
axs[1].set_xlabel("$k$")
axs[1].set_ylabel("Max error")
axs[1].legend()

plt.show()