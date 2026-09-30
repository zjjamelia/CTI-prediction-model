import numpy as np


def diffusionRWR(A, max_iter, restart_prob):
    n = A.shape[0]

    # Add self-edge to isolated nodes
    A = A + np.diag(np.sum(A, axis=1) == 0)

    # Normalize the adjacency matrix
    P = A / np.sum(A, axis=1, keepdims=True)

    # Personalized PageRank
    restart = np.eye(n)
    Q = np.eye(n)

    for i in range(max_iter):
        Q_new = (1 - restart_prob) * P @ Q + restart_prob * restart
        delta = np.linalg.norm(Q - Q_new, 'fro')
        Q = Q_new
        if delta < 1e-6:
            break

    return Q
