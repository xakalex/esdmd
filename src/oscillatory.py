import numpy as np


class Oscillatory:
    def __init__(self, n, dt, duration, rng, params):
        self.n = n
        self.duration = duration
        self.dt = dt
        self.f1 = params["f1"]
        self.f2 = params["f2"]
        self.noise_cov = params["noise_cov"]
        self.rng = rng
        v_dist = params["v_dist"]
        self.v1, self.v2, self.v3, self.v4 = [rng.normal(v_dist["mean"], v_dist["std"], n) for _ in range(4)]

        self.t, self.y = self.simulate()

    def __call__(self):
        return self.t, self.y

    def ode(self, k):
        return (
            (self.v1 * np.cos(2 * np.pi * self.dt * self.f1 * k))
            + (self.v2 * np.cos(2 * np.pi * self.dt * self.f2 * k))
            + (self.v3 * np.sin(2 * np.pi * self.dt * self.f1 * k))
            + (self.v4 * np.sin(2 * np.pi * self.dt * self.f2 * k))
        )

    def simulate(self):
        t = np.linspace(0, self.duration, int(self.duration / self.dt))
        Y = np.zeros((self.n, len(t)))
        for idx, k in enumerate(range(len(t))):
            vec = self.ode(k)
            Y[:, idx] = vec + (np.sqrt(self.noise_cov) * self.rng.normal(size=self.n))
        return t, Y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    duration = 15
    dt = 0.01

    params = dict(
        v_dist={"mean": 0.0, "std": 1.0},  # distribution for the oscillatory system
        f1=5.2,  # frequency component 1 for the oscillatory system
        f2=1.0,  # frequency component 2 for the oscillatory system
        noise_cov=1e-3,  # noise covariance for the oscillatory system
    )

    system = Oscillatory(n, dt, duration, rng, params)
    t, y = system()

    # plot simulated system
    import matplotlib.pyplot as plt

    for i in range(n):
        plt.plot(t, y[i])

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Oscillatory System Simulation")
    plt.show()
