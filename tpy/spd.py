import numpy as np
import scipy as sp


def rand_cov(n=5):
    S = np.random.randn(n, n)
    return S.dot(S.T)


def spd_root(S):
    L, Q = np.linalg.eig(S)
    return Q.dot(np.diag(L**0.5)).dot(Q.T)


def cov_to_std(S):
    return np.diag(np.diag(S)**0.5)


def cov_to_std_corr(S):
    std = cov_to_std(S)
    std_1 = np.linalg.inv(std)
    R = std_1.dot(S).dot(std_1)
    return std, R


def cov_to_corr(S):
    return cov_to_std_corr(S)[1]



