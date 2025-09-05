import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Lorenz96:
    def __init__(self, n, dt, duration, rng, params):
        self.n = n
        self.dt = dt
        self.duration = duration
        self.F = params.get("F", 8.0)
        self.x0 = rng.normal(0.0, params.get("x0_std", 1.0), n)
        self.t, self.y = self.simulate()

    def __call__(self):
        return self.t, self.y

    def derivative(self, x, t):
        xp1 = np.roll(x, -1)
        xm1 = np.roll(x, 1)
        xm2 = np.roll(x, 2)
        return (xp1 - xm2) * xm1 - x + self.F

    def integrate(self, x0, t):
        Y = odeint(self.derivative, x0, t)  # (T, n)
        return Y.T

    def simulate(self):
        t = np.linspace(0, self.duration, int(self.duration / self.dt))
        Y = self.integrate(self.x0, t)
        return t, Y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 50
    duration = 10
    dt = 0.01

    params = dict(
        F=8.0,  # Forcing
        x0_std=1.0,  # init stddv
    )

    system = Lorenz96(n, dt, duration, rng, params)
    t, y = system()

    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(t, y[i])
    plt.xlabel("Time")
    plt.ylabel("State variables")
    plt.title("Lorenz '96")
    plt.tight_layout()
    plt.show()
