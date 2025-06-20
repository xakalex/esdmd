import numpy as np
from networkx import erdos_renyi_graph, to_numpy_array
from scipy.integrate import odeint


class Kuramoto:
    def __init__(self, n, dt, duration, rng, params):
        self.n = n
        self.duration = duration
        self.dt = dt
        self.theta0 = rng.uniform(params["irange"][0], params["irange"][1], n)
        self.K = rng.uniform(params["K_range"][0], params["K_range"][1])
        self.omega = rng.uniform(params["omega_range"][0], params["omega_range"][1], n)
        self.adj_mat = to_numpy_array(erdos_renyi_graph(n, params["adj_prob"], seed=rng))
        self.gamma = params["gamma"]

        self.t, self.y = self.simulate()

    def __call__(self):
        return self.t, np.sin(self.y)

    def derivative(self, angles_vec, t):
        angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
        interactions = self.adj_mat * np.sin(angles_j - angles_i)  # Aij * sin(j-i)
        # dxdt = (self.omega / (1 + self.gamma)) + (self.K / (1 + self.gamma)) * interactions.sum(axis=0)
        dxdt = self.omega + self.K * interactions.sum(axis=0)  # sum over incoming interactions
        dxdt -= self.gamma * dxdt
        return dxdt

    def integrate(self, angles_vec, t):
        """Updates all states by integrating state of all nodes"""
        timeseries = odeint(self.derivative, angles_vec, t)
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def simulate(self):
        angles_vec = self.theta0
        t = np.linspace(0, self.duration, int(self.duration / self.dt))
        Y = self.integrate(angles_vec, t)

        return t, Y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    duration = 15
    dt = 0.01

    params = dict(
        irange=(0.0, 2 * np.pi),  # range for initial conditions (0 to 2*pi)
        omega_range=(2.5, 3.5),  # range for natural frequencies
        K_range=(0.1, 0.2),  # coupling strength range
        adj_prob=0.9,  # probability of connection in the Kuramoto network
        gamma=0.1,  # damping factor for the Kuramoto system
    )

    system = Kuramoto(n, dt, duration, rng, params)
    t, y = system()

    # plot simulated system
    import matplotlib.pyplot as plt

    for i in range(n):
        plt.plot(t, np.sin(y[i]))
    plt.xlabel("Time")
    plt.ylabel("Sin of Oscillator Phase")
    plt.title("Kuramoto System Simulation")
    plt.show()
