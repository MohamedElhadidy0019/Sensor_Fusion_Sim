

import numpy as np


class DistributedFilter:
    """N persistent local sensor tracks, globalized every step before
    prediction, fused via the information filter.
    """

    def __init__(self, n_sensors, F, H, Q, R, x0, P0):
        self.n_sensors = n_sensors
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        # Fusion Center's own persistent central (globally-fused) estimate.
        self.x = x0.copy()
        self.P = P0.copy()

        # One independent, persistent local track per sensor.
        self.x_local = [x0.copy() for _ in range(n_sensors)]
        self.P_local = [P0.copy() for _ in range(n_sensors)]

    def _fuse(self, x_list, P_list):
        """Combine a list of (x, P) estimates via inverse-covariance
        weighting (convex combination).
        """
        inner_x_sum = np.zeros_like(x_list[0])
        inner_p_sum = np.zeros_like(P_list[0])
        for x, P in zip(x_list, P_list):
            P_inv = np.linalg.inv(P)
            inner_x_sum += P_inv @ x
            inner_p_sum += P_inv
        P_fused = np.linalg.inv(inner_p_sum)
        x_fused = P_fused @ inner_x_sum

        return x_fused, P_fused

    def _globalize(self):

        x_global, P_global = self._fuse(self.x_local, self.P_local)
        P_tilde = self.n_sensors * P_global

        x_tilde = [P_tilde @ np.linalg.inv(P_s) @ x_s
                   for x_s, P_s in zip(self.x_local, self.P_local)]

        return x_tilde, P_tilde

    def _predict_update(self, s, z, x_tilde_s, P_tilde):
        """Local KF predict (from sensor s's globalized state) + update,
        given its measurement z.
        """
        F, H, R = self.F, self.H, self.R
        Q_inflated = self.n_sensors * self.Q

        # Predict -- from the globalized state, shared inflated covariance.
        x_pred = F @ x_tilde_s
        P_pred = F @ P_tilde @ F.T + Q_inflated

        # Update
        nu = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        W = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + W @ nu
        P_upd = P_pred - W @ S @ W.T

        self.x_local[s] = x_upd
        self.P_local[s] = P_upd

    def step(self, measurements):
        """Advance one step: globalize, predict+update every local filter
        from its globalized state, then re-fuse.

        measurements: list of S measurement vectors z_1..z_S.
        Returns (x_fused, P_fused).
        """
        x_tilde, P_tilde = self._globalize()

        for s, z in enumerate(measurements):
            self._predict_update(s, z, x_tilde[s], P_tilde)

        self.x, self.P = self._fuse(self.x_local, self.P_local)
        return self.x, self.P


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from convex_comb import ConvexCombinationFusion
    from sim import Sim, cv_model

    N = 50
    n_sensors = 4

    sim = Sim(n_sensors=n_sensors, seed=0)
    F, H, Q, R = cv_model()

    # Uninformative prior: filters don't get to "know" the true start state.
    x0 = np.zeros(4)
    P0 = 1000.0 * np.eye(4)

    distributed = DistributedFilter(n_sensors, F, H, Q, R, x0, P0)
    convex = ConvexCombinationFusion(n_sensors, F, H, Q, R, x0, P0)

    truths = []
    distributed_track = []
    convex_track = []

    for _ in range(N):
        truth, measurements = sim.step()

        # Both methods see the exact same measurements, for a fair comparison.
        x_distributed, _ = distributed.step(measurements)
        x_convex, _ = convex.step(measurements)

        truths.append(truth)
        distributed_track.append(x_distributed)
        convex_track.append(x_convex)

    truths = np.array(truths)                        # (N, 4)
    distributed_track = np.array(distributed_track)   # (N, 4)
    convex_track = np.array(convex_track)             # (N, 4)

    # Position RMSE over the whole run: sqrt(mean over time of squared
    # Euclidean distance between estimate and ground truth).
    rmse_distributed = np.sqrt(np.mean(np.sum((distributed_track[:, :2] - truths[:, :2])**2, axis=1)))
    rmse_convex = np.sqrt(np.mean(np.sum((convex_track[:, :2] - truths[:, :2])**2, axis=1)))
    print(f"RMSE (distributed KF):       {rmse_distributed:.3f}")
    print(f"RMSE (convex combination):   {rmse_convex:.3f}")

    plt.figure(figsize=(8, 8))

    plt.plot(convex_track[:, 0], convex_track[:, 1], "c-o", markersize=4,
              linewidth=2, label="Fused track (convex combination)")
    plt.plot(distributed_track[:, 0], distributed_track[:, 1], "m-o", markersize=4,
              linewidth=2, label="Fused track (distributed KF)")

    plt.plot(truths[:, 0], truths[:, 1], "k-o", markersize=4,
              linewidth=2, label="Ground truth")
    plt.scatter(truths[0, 0], truths[0, 1], s=80, color="black",
                marker="^", zorder=5, label="Start")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Distributed KF vs. convex combination vs. ground truth")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
