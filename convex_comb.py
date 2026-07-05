"""Track-to-Track fusion via naive convex combination.

Each of the S sensors runs its own local Kalman Filter (predict + update),
completely independently, with no communication between sensors. The Fusion
Center then combines the S local posteriors at every step by
inverse-covariance weighting (convex combination).

This ignores the cross-covariance between the local estimates, so it is only
exact when the process noise Q = 0 (independent sensor errors). It's the
"naive" baseline fusion method.
"""

import numpy as np


class ConvexCombinationFusion:
    """S independent local KFs + convex-combination Fusion Center."""

    def __init__(self, n_sensors, F, H, Q, R, x0, P0):
        self.n_sensors = n_sensors
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        # One independent (x, P) per sensor.
        self.x = [x0.copy() for _ in range(n_sensors)]
        self.P = [P0.copy() for _ in range(n_sensors)]

    def _predict_update(self, s, z):
        """Local KF predict + update for sensor s, given its measurement z."""
        F, H, Q, R = self.F, self.H, self.Q, self.R

        # Predict
        x_pred = F @ self.x[s]
        P_pred = F @ self.P[s] @ F.T + Q

        # Update
        nu = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        W = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + W @ nu
        P_upd = P_pred - W @ S @ W.T

        self.x[s] = x_upd
        self.P[s] = P_upd

    def _convex_combination(self):
        """Combine self.x[s], self.P[s] for s in range(n_sensors) into a
        single fused (x_fused, P_fused) via inverse-covariance weighting.
        """
        inner_x_sum = np.zeros_like(self.x[0])
        inner_p_sum = np.zeros_like(self.P[0])
        for s in range(self.n_sensors):
            P_inv = np.linalg.inv(self.P[s])
            inner_x_sum += P_inv @ self.x[s]
            inner_p_sum += P_inv
        P_fused = np.linalg.inv(inner_p_sum)
        x_fused = P_fused @ inner_x_sum
        
        return x_fused, P_fused
        

    def step(self, measurements):
        """Advance one step: predict+update every local filter, then fuse.

        measurements: list of S position measurements z_1..z_S.
        Returns (x_fused, P_fused).
        """
        for s, z in enumerate(measurements):
            self._predict_update(s, z)

        return self._convex_combination()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from sim import Sim, cv_model

    N = 50
    n_sensors = 30

    sim = Sim(n_sensors=n_sensors, seed=0)
    F, H, Q, R = cv_model()

    # Uninformative prior: filters don't get to "know" the true start state.
    x0 = np.zeros(4)
    P0 = 1000.0 * np.eye(4)

    fusion = ConvexCombinationFusion(n_sensors, F, H, Q, R, x0, P0)

    truths = []
    local_tracks = [[] for _ in range(n_sensors)]
    fused_track = []

    for _ in range(N):
        truth, measurements = sim.step()
        x_fused, P_fused = fusion.step(measurements)

        truths.append(truth)
        fused_track.append(x_fused)
        for s in range(n_sensors):
            local_tracks[s].append(fusion.x[s].copy())

    truths = np.array(truths)              # (N, 4)
    fused_track = np.array(fused_track)    # (N, 4)

    plt.figure(figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_sensors))
    for s in range(n_sensors):
        track = np.array(local_tracks[s])  # (N, 4)
        plt.plot(track[:, 0], track[:, 1], "--", color=colors[s], alpha=0.6,
                  label=f"Sensor {s+1} local track")

    plt.plot(fused_track[:, 0], fused_track[:, 1], "m-o", markersize=4,
              linewidth=2, label="Fused track (convex combination)")

    plt.plot(truths[:, 0], truths[:, 1], "k-o", markersize=4,
              linewidth=2, label="Ground truth")
    plt.scatter(truths[0, 0], truths[0, 1], s=80, color="black",
                marker="^", zorder=5, label="Start")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Track-to-Track fusion: local tracks vs. fused vs. ground truth")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
