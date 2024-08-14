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
    start_time = time.time()
    naive_output = calculate_attention_batched(Q, K, V)
    print("Naive time = ", time.time() - start_time)
    # print(naive_output)

    # Call the attn_forward function.
    start_time = time.time()
    attn_output = attn_forward_batched(Q, K, V, k=k)
    print("Attn time = ", time.time() - start_time)
    # print(attn_output)

    return torch.mean(torch.abs(naive_output - attn_output))

torch.manual_seed(0)
b = 1
h = 10
n, d = 1000, 32

# Range k.
kk = [int(n ** 0.5), int(n ** 0.25), int(n ** 0.125), int(math.log(n)), 3]
kk = sorted(list(set(kk)))
kk_labels = [r"2", r"$\log(n)$", r"$n^{1/8}$", r"$n^{1/4}$", r"$n^{1/2}$"]
if len(kk_labels) != len(kk):
    print(kk)
    print(kk_labels)
    raise ValueError("kk_labels and kk must have the same length.")

# Range B.
BB = [2, 2.5, 3, 3.5, 4]
colors = ['b', 'g', 'r', 'c', 'm']

for i, B in enumerate(BB):
    means = []
    std_devs = []
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

        print("Q shape = ", Q.shape)

        num_iterations = 10
        
        results_k = [run_experiment(Q, K, V, k=k) for _ in range(num_iterations)]
        means.append(torch.mean(torch.tensor(results_k)))
        print("k = ", k, " mean = ", means[-1])
        std_devs.append(torch.std(torch.tensor(results_k)))
        print("k = ", k, " std_dev = ", std_devs[-1])

    means = np.array(means)
    std_devs = np.array(std_devs)

    plt.plot(kk_labels, means, 'b-o', color=colors[i], label=f"B={B}")
    plt.fill_between(kk_labels, means - std_devs, means + std_devs, color='b', alpha=0.2)

plt.title("Mean Approximation Error of kNN Attention")
plt.xlabel("$k$")
plt.ylabel("Mean error")
plt.legend()


plt.show()