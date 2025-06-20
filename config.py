import numpy as np


class Config:
    def __init__(self):
        # system generation configuration
        self.sysname = "oscillatory"  # name of the dynamical system to run: kuramoto or oscillatory
        self.n = 100  # number of states to simulate
        self.fs = 120  # sampling frequency in Hz
        self.duration = 10  # total duration of the simulation in seconds
        self.rng = np.random.default_rng()

        # dmd configuration
        self.streaming_rank = 10  # rank used by the streaming algorithm
        self.batch_rank = self.n  # rank used by the batch algorithm

        # system specific configuration
        self.sys_params = dict(
            kuramoto=dict(
                irange=(0.0, 2 * np.pi, self.n),  # initial conditions
                omega_range=(2.5, 3),  # range for natural frequencies
                K_range=(0.1, 0.2),  # coupling strength range
                adj_prob=0.99,  # probability of connection in the Kuramoto network
                gamma=0.9,  # damping factor for the Kuramoto system
            ),
            oscillatory=dict(
                v_dist=dict(mean=0.0, std=1.0),  # distribution for the oscillatory system
                noise_cov=1e-3,  # noise covariance
                f1=5.2,  # frequency component 1
                f2=1.0,  # frequency component 2
            ),
        )

        # plotting configuration
        self.markerstyle = dict(esdmd="D", sdmd="p", dmd="o")
        self.markercolor = dict(esdmd="red", sdmd="blue", dmd="grey")

        # plot display
        self.show_plots = {"eigenvalues"}  # all, mode-frequency, exectime, eigenvalues
