import numpy as np
import os
import scipy.io as sio


def diffusionRWR(A, max_iter, restart_prob):
    #A是图的邻接矩阵
    #n是图中的节点数量
    n = A.shape[0]
    #加一些0的边保证图连通性
    # Add self-edge to isolated nodes
    A = A + np.diag(np.sum(A, axis=1) == 0)
    #归一化为概率矩阵P
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


def joint(networks, rsp, max_iter):
    Q_list = []
    print(f'{networks}.txt')
    for network in networks:
        file_path = os.path.join('../network', f'{network}.txt')
        net = np.loadtxt(file_path)
        tQ = diffusionRWR(net, max_iter, rsp)
        Q_list.append(tQ)

    Q = np.hstack(Q_list)

    nnode = Q.shape[0]
    alpha = 1 / nnode
    Q = np.log(Q + alpha) - np.log(alpha)

    return Q