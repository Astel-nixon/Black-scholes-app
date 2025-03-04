import numpy as np

def binomial_tree_call(S, K, T, r, sigma, steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    option_tree = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        option_tree[i, steps] = max(S * (u ** (steps - i)) * (d ** i) - K, 0)
    
    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            option_tree[i, j] = np.exp(-r * dt) * (p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1])
    
    return option_tree[0, 0]
