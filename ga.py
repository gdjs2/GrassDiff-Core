import numpy as np
import torch
import pygmtools as pygm
from numba import jit

pygm.BACKEND = 'pytorch'


def graduate_assignment(G, g, sim, beta_0=50, beta_f=1000, beta_r=2, I0=4, I1=30, eps=0.005, gt=None, threshold=0.2, type_info=None):
    G = G.to(sim.device)
    g = g.to(sim.device)
    beta = beta_0
    sim = sim.clone()
    sim[sim < threshold] = 0
    M0 = sim
    while beta < beta_f:
        M0_history = []
        for i in range(I0):
            Q = _differential2(M0, G, g, sim)
            M0 = beta * Q * sim
            # M0 = beta * Q
            # M0_new = pygm.sinkhorn(M0,
            #                        max_iter=I1,
            #                        batched_operation=True,
            #                        # unmatch1=torch.zeros(M0.size(0), device=M0.device, dtype=M0.dtype),
            #                        # unmatch2=torch.zeros(M0.size(1), device=M0.device, dtype=M0.dtype)
            #                        )
            M0_new = pygm.hungarian(M0)
            if type_info is not None:
                t_mask, s_pairs = type_info
                if len(s_pairs) != 0:
                    M0_new[t_mask] = 0
                    M0_new[s_pairs[0], :] = 0
                    M0_new[:, s_pairs[1]] = 0
                    M0_new[s_pairs[0], s_pairs[1]] = 1
            M0 = M0_new
            M0_history.append(M0)
        diff1 = torch.max(torch.abs(M0_history[-1] - M0_history[0]))
        diff2 = torch.max(torch.abs(M0_history[-1] - M0_history[1]))
        diff3 = torch.max(torch.abs(M0_history[-1] - M0_history[2]))
        if beta > 100 and (diff1 < eps or diff2 < eps or diff3 < eps):
            # print('early break!!')
            break
        beta *= beta_r
    if gt is not None:
        n1, n2 = gt.size()
        m = _differential2(M0[:n1, :n2], G[:n1, :n1], g[:n2, :n2], sim[:n1, :n2])
        ms = m.sum()
        e = _differential2(gt, G[:n1, :n1], g[:n2, :n2], sim[:n1, :n2])
        es = e.sum()
    return M0


def _differential2(M0, G, g, sim):
    # Q = G @ (M0 * sim) @ g.t() + 2 * sim
    # Q = G @ M0 @ g.t() + sim * 2
    Q = torch.linalg.multi_dot((G, M0 * sim, g.t())) + 2 * sim
    return Q


@jit(nopython=True)
def compute_q(A, I, Q, M0, G, g, sim):
    for a in range(A):
        for i in range(I):
            for b in range(A):
                for j in range(I):
                    Q[a, i] += M0[b, j] * G[a, b] * g[i, j]
            Q[a, i] += 2 * sim[a, i]


def ga_original(G, g, sim, beta_0=0.5, beta_f=10, beta_r=1.075, I0=4, I1=30, eps=0.005, gt=None, threshold=0.2, type_info=None):
    G = G.to(sim.device)
    g = g.to(sim.device)
    beta = beta_0
    sim = sim.clone()
    sim[sim < threshold] = 0
    M0 = torch.ones_like(sim)
    M0 = sim
    while beta < beta_f:
        for i in range(I0):
            Q = G @ M0 @ g.t() + sim * 2
            M0 = beta * Q
            M0_new = pygm.hungarian(M0)
            if torch.abs(M0_new - M0).sum() < 0.05:
                print('break')
                beta = beta_f * 100
                break
            M0 = M0_new
        beta *= beta_r
    return M0


def clean_up_heuristic(M0):
    max_val, index = torch.max(M0, dim=1)
    mask = max_val >= 0.4
    result = torch.zeros_like(M0)
    idx = torch.arange(M0.size(0), dtype=torch.int64, device=M0.device)
    result[idx[mask], index[mask]] = 1
    return result
