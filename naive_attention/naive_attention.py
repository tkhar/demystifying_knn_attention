import torch

def multiply(Q, K):
    # Q -> (n, d)
    # K -> (d, n)
    # Result -> (n, n)
    result = torch.zeros(Q.size(0), K.size(1))
    for i in range(Q.size(0)):
        for j in range(K.size(1)):
            result[i, j] = torch.dot(Q[i,:], K[:,j])

    return result
            

def calculate_attention(Q, K, V, B):
    n, d = Q.size()

    # Calculate the dot product of Q and K
    attention_scores = multiply(Q, K.transpose(-2, -1))

    # Normalize the attention scores by dividing by d.
    attention_scores = attention_scores / d

    # Numerical stability
    attention_scores -= B*B
    
    # Apply softmax to get the attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)
    
    # Multiply the attention weights with the value vectors
    attention_output = torch.matmul(attention_weights, V)
    
    return attention_output