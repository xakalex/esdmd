import copy
from time import perf_counter as timer

import numpy as np

from src.esdmd import EfficientStreamingDMD
from src.kuramoto import Kuramoto
from src.lorenz96 import Lorenz96
from src.oscillatory import Oscillatory
from src.sdmd import StreamingDMD


class DataUtils:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = 1 / cfg.fs
        self.m = (cfg.duration * cfg.fs) - 1

        self.sys_objects = dict(kuramoto=Kuramoto, oscillatory=Oscillatory, lorenz96=Lorenz96)
        self.t, self.data_matrix = self.simulate_system()
        self.X, self.Y = self.data_matrix[:, :-1], self.data_matrix[:, 1:]

        self.dmd_modes, self.dmd_evals = self.compute_dmd_modes()

        self.sdmd = StreamingDMD(self.cfg.n, self.cfg.streaming_rank)
        self.esdmd = EfficientStreamingDMD(self.cfg.n, self.cfg.streaming_rank)

        self.sdmd_modes, self.sdmd_evals = [], np.zeros((self.cfg.streaming_rank, self.m), dtype=complex)
        self.esdmd_modes, self.esdmd_evals = [], np.zeros((self.cfg.streaming_rank, self.m), dtype=complex)
        self.sdmd_diff_ms, self.esdmd_diff_ms = [], []

    @staticmethod
    def streaming_step(alg, x, y):
        s = timer()
        alg.update(x, y)
        e = timer()
        diff_ms = (e - s) * 1000

        modes, evals = alg.compute_modes()
        evals = evals.reshape(-1, 1)

        return modes, evals, diff_ms

    def dynamic_resize(self, modes, evals):
        missing = self.cfg.streaming_rank - evals.shape[0]
        if missing > 0:
            eval_padding = np.zeros((missing, 1), dtype=complex)
            modes_padding = np.zeros((self.cfg.n, missing), dtype=complex)
            evals = np.concatenate((evals, eval_padding), axis=0)
            modes = np.concatenate((modes, modes_padding), axis=1)
        return modes, evals

    def run_streaming_algorithms(self, k):
        x, y = copy.deepcopy(self.X[:, k]).reshape(-1, 1), copy.deepcopy(self.Y[:, k]).reshape(-1, 1)

        esdmd_modes_, esdmd_evals_, esdmd_diff_ms = self.streaming_step(self.esdmd, x, y)
        esdmd_modes, esdmd_evals = self.dynamic_resize(esdmd_modes_, esdmd_evals_)

        sdmd_modes_, sdmd_evals_, sdmd_diff_ms = self.streaming_step(self.sdmd, x, y)
        sdmd_modes, sdmd_evals = self.dynamic_resize(sdmd_modes_, sdmd_evals_)

        self.esdmd_modes.append(esdmd_modes)
        self.esdmd_evals[:, k] = esdmd_evals.reshape(-1)
        self.esdmd_diff_ms.append(esdmd_diff_ms)

        self.sdmd_modes.append(sdmd_modes)
        self.sdmd_evals[:, k] = sdmd_evals.reshape(-1)
        self.sdmd_diff_ms.append(sdmd_diff_ms)

    def simulate_system(self):
        system = self.sys_objects.get(self.cfg.sysname)
        return system(self.cfg.n, self.dt, self.cfg.duration, self.cfg.rng, self.cfg.sys_params[self.cfg.sysname])()

    def compute_dmd_modes(self):
        U, s, Vh = np.linalg.svd(self.X, full_matrices=False)
        r = self.cfg.batch_rank
        Ur = U[:, :r]
        sr = s[:r]
        Vhr = Vh[:r, :]

        A_tilde = Ur.conj().T @ self.Y @ Vhr.conj().T @ np.diag(1.0 / sr)
        evals, evecs = np.linalg.eig(A_tilde)

        dmd_modes = Ur @ evecs

        return dmd_modes, evals
