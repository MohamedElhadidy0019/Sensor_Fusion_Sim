"""Simulation of a constant-velocity target observed by S sensors.

State vector (per the assignment, slide 39):

    x = [px, py, vx, vy]^T

Each sensor measures the position only:

    z = [px, py]^T + v,   v ~ N(0, R),   R = r * I_2

The Sim owns the *ground truth* and produces, at every time step, one noisy
measurement per sensor.  It knows nothing about the fusion methods.
"""

import numpy as np


def cv_model(dt=1.0, q=0.1, r=100.0):
    """Return the constant-velocity model matrices (F, H, Q, R).

    F  state transition          (4x4)
    H  measurement matrix         (2x4)  -> observes position only
    Q  process-noise covariance   (4x4)  -> discrete white-noise acceleration
    R  measurement-noise cov.     (2x2)  -> r * I_2
    """
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])

    # Discrete white-noise acceleration model, q = noise spectral density.
    Q = q * np.array([
        [dt**3 / 3, 0.0,       dt**2 / 2, 0.0      ],
        [0.0,       dt**3 / 3, 0.0,       dt**2 / 2],
        [dt**2 / 2, 0.0,       dt,        0.0      ],
        [0.0,       dt**2 / 2, 0.0,       dt       ],
    ])

    R = r * np.eye(2)
    return F, H, Q, R


class Sim:
    """Ground-truth target + noisy per-sensor measurements."""

    def __init__(self, n_sensors=4, dt=1.0, q=0.1, r=100.0, x0=None, seed=0):
        self.n_sensors = n_sensors
        self.F, self.H, self.Q, self.R = cv_model(dt, q, r)
        self.x_true = np.array([0.0, 0.0, 1.0, 1.0]) if x0 is None else np.asarray(x0, float)
        self.rng = np.random.default_rng(seed)

    def step(self):
        """Advance the truth one step and return (truth, [z_1, ..., z_S])."""
        # Move the target forward with process noise.
        process_noise = self.rng.multivariate_normal(np.zeros(4), self.Q)
        self.x_true = self.F @ self.x_true + process_noise

        # Each sensor sees the position plus its own measurement noise.
        measurements = []
        for _ in range(self.n_sensors):
            meas_noise = self.rng.multivariate_normal(np.zeros(2), self.R)
            measurements.append(self.H @ self.x_true + meas_noise)

        return self.x_true.copy(), measurements


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 50
    sim = Sim(n_sensors=4, seed=0)

    truths = []
    all_meas = [[] for _ in range(sim.n_sensors)]

    for _ in range(N):
        truth, measurements = sim.step()
        truths.append(truth)
        for s, z in enumerate(measurements):
            all_meas[s].append(z)

    truths = np.array(truths)          # (N, 4)

    plt.figure(figsize=(8, 8))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for s in range(sim.n_sensors):
        zs = np.array(all_meas[s])    # (N, 2)
        plt.scatter(zs[:, 0], zs[:, 1], s=12, alpha=0.5,
                    color=colors[s], label=f"Sensor {s+1} measurements")

    plt.plot(truths[:, 0], truths[:, 1], "k-o", markersize=4,
             linewidth=2, label="Ground truth")
    plt.scatter(truths[0, 0], truths[0, 1], s=80, color="black",
                marker="^", zorder=5, label="Start")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Simulation: ground truth + raw sensor measurements")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
