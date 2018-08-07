import numpy as np
from .spd import spd_root


def w2_cost(P, Q, P2=None):
    if P2 is None:
        P2 = spd_root(P)
    return np.sum(np.abs(np.diag(P + Q -2*spd_root(P2.dot(Q).dot(P2)) )))**0.5


def w2_map(P, Q):
    P2 = spd_root(P)
    P_2 = np.linalg.inv(P2)
    return P_2.dot(spd_root(P2.dot(Q).dot(P2))).dot(P_2)


def w2_barycenter_risk(k0, W, K):
    k2 = spd_root(k0)
    r = 0
    for wi, ki in zip(W, K):
        r += wi*w2_cost(k0, ki, k2)**2
    return r


def w2_barycenter_step_slow(k0, W, K):
    k2 = spd_root(k0)
    r = 0
    for i, (wi,ki) in enumerate(zip(W,K)):
        k2i = spd_root(k2.dot(ki).dot(k2))
        r += wi*k2i
    return r


def w2_barycenter_step_fast(k0, W, K):
    k2 = spd_root(k0)
    k2inv = np.linalg.inv(k2)
    r = 0
    for i, (wi,ki) in enumerate(zip(W,K)):
        k2i = spd_root(k2.dot(ki).dot(k2))
        r += wi*k2i
    return k2inv.dot(r.dot(r)).dot(k2inv)