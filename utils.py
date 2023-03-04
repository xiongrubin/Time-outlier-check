import numpy as np


def power_method(A, iter_num=1):

    # set initial vector q
    q = np.random.normal(size=A.shape[1])
    q = q / np.linalg.norm(q)

    for i in range(iter_num):
        q = np.dot(np.dot(A.T, A), q)

    v = q / np.linalg.norm(q)
    Av = np.dot(A, v)
    s = np.linalg.norm(Av)
    u = Av / s

    return u, s, v

def tridiagonalize_by_lanczos(P, m, k):

    # Initialize variables
    T = np.zeros((k, k))
    r0 = m
    beta0 = 1
    q0 = np.zeros(m.shape)

    for i in range(k):
        q1 = r0 / beta0
        C = np.dot(P, q1)
        alpha1 = np.dot(q1, C)
        r1 = C - alpha1 * q1 - beta0 * q0
        beta1 = np.linalg.norm(r1)

        T[i, i] = alpha1
        if i + 1 < k:
          T[i, i + 1] = beta1
          T[i + 1, i] = beta1

        q0 = q1
        beta0 = beta1
        r0 = r1

    return T

def tridiag_eigen(T, iter_num=1, tol=1e-3):

    eigenvectors = np.identity(T.shape[0])

    for i in range(iter_num):
        Q, R = tridiag_qr_decomposition(T)
        T = np.dot(R, Q)
        eigenvectors = np.dot(eigenvectors, Q)
        eigenvalue = np.diag(T)
        if np.all((T - np.diag(eigenvalue) < tol)):
            break

    return eigenvalue, eigenvectors

def tridiag_qr_decomposition(T):

    R = T.copy()
    Qt = np.eye(T.shape[0])

    for i in range(T.shape[0] - 1):
        u = householder(R[i:i + 2, i])
        M = np.outer(u, u)
        R[i:i + 2, :(i + 3)] -= 2 * np.dot(M, R[i:i + 2, :(i + 3)])
        Qt[i:i + 2, :(i + 3)] -= 2 * np.dot(M, Qt[i:i + 2, :(i + 3)])

    return Qt.T, R

def householder(x):

    x[0] = x[0] + np.sign(x[0]) * np.linalg.norm(x)
    x = x / np.linalg.norm(x)
    return x
