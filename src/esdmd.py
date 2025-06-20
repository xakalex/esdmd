import numpy as np
from scipy.linalg import pinvh


class EfficientStreamingDMD:
    def __init__(self, n, rank, n_gram=5, eps=1e-10):
        self.n = n
        self.rank = rank
        self.n_gram = n_gram
        self.eps = eps
        self.first = True

    def update(self, x, y):
        newx = False
        if self.first:
            Q, _ = np.linalg.qr(np.column_stack((x, y)))
            self.Q = Q[:, : min(self.rank, Q.shape[1])]  # accounting for rank = 1
            xtilde = self.Q.T @ x
            ytilde = self.Q.T @ y
            self.Gx = xtilde @ xtilde.T
            self.Gy = ytilde @ ytilde.T
            self.A = ytilde @ xtilde.T
            self.last_ytilde = ytilde
            self.first = False
            return newx

        ey = y.copy()
        for _ in range(self.n_gram):
            dy = self.Q.T @ ey
            ey -= self.Q @ dy

        normey = np.linalg.norm(ey)
        if normey / np.linalg.norm(y) > self.eps:
            self.Q = np.concatenate((self.Q, ey / normey), axis=1)
            self.Gy = np.block([[self.Gy, np.zeros((self.Gy.shape[0], 1))], [np.zeros((1, self.Gy.shape[1] + 1))]])
            self.Gx = np.block([[self.Gx, np.zeros((self.Gx.shape[0], 1))], [np.zeros((1, self.Gx.shape[1] + 1))]])
            self.A = np.block([[self.A, np.zeros((self.A.shape[0], 1))], [np.zeros((1, self.A.shape[1] + 1))]])
            if self.Q.shape[1] > self.rank:
                evals, evecs = np.linalg.eigh(self.Gy)
                idx = evals.argsort()[::-1][: self.rank]
                qy = evecs[:, idx]
                self.Q = self.Q @ qy
                self.A = qy.T @ self.A @ qy
                self.Gx = qy.T @ self.Gx @ qy
                self.Gy = np.diag(evals[idx])
            newx = True

        ytilde = self.Q.T @ y
        xtilde = self.Q.T @ x if newx else self.last_ytilde
        self.Gx += xtilde @ xtilde.T
        self.Gy += ytilde @ ytilde.T
        self.A += ytilde @ xtilde.T
        self.last_ytilde = ytilde
        return newx

    def compute_modes(self):
        evals, evecs = np.linalg.eig(self.A @ pinvh(self.Gx))
        modes = self.Q @ evecs
        return modes, evals
