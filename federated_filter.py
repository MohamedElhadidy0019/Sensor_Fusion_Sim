
import numpy as np


class FederatedFilter:
    """S independent local KFs (inflated Q) + convex-combination Fusion
    Center.
    """

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
        """Local KF predict (inflated Q) + update for sensor s, given its
        measurement z.
        """
        F, H, R = self.F, self.H, self.R
        Q_inflated = self.n_sensors * self.Q

        # Predict -- inflated process noise, deck6 eq. 528.
        x_pred = F @ self.x[s]
        P_pred = F @ self.P[s] @ F.T + Q_inflated

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

        measurements: list of S measurement vectors z_1..z_S.
        Returns (x_fused, P_fused).
        """
        for s, z in enumerate(measurements):
            self._predict_update(s, z)

        return self._convex_combination()


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

    federated = FederatedFilter(n_sensors, F, H, Q, R, x0, P0)
    convex = ConvexCombinationFusion(n_sensors, F, H, Q, R, x0, P0)

    truths = []
    federated_track = []
    convex_track = []

    for _ in range(N):
        truth, measurements = sim.step()

        # Both methods see the exact same measurements, for a fair comparison.
        x_federated, _ = federated.step(measurements)
        x_convex, _ = convex.step(measurements)

        truths.append(truth)
        federated_track.append(x_federated)
        convex_track.append(x_convex)

    truths = np.array(truths)                    # (N, 4)
    federated_track = np.array(federated_track)   # (N, 4)
    convex_track = np.array(convex_track)         # (N, 4)

    # Position RMSE over the whole run: sqrt(mean over time of squared
    # Euclidean distance between estimate and ground truth).
    rmse_federated = np.sqrt(np.mean(np.sum((federated_track[:, :2] - truths[:, :2])**2, axis=1)))
    rmse_convex = np.sqrt(np.mean(np.sum((convex_track[:, :2] - truths[:, :2])**2, axis=1)))
    print(f"RMSE (federated KF):         {rmse_federated:.3f}")
    print(f"RMSE (convex combination):   {rmse_convex:.3f}")

    plt.figure(figsize=(8, 8))

    plt.plot(convex_track[:, 0], convex_track[:, 1], "c-o", markersize=4,
              linewidth=2, label="Fused track (convex combination)")
    plt.plot(federated_track[:, 0], federated_track[:, 1], "m-o", markersize=4,
              linewidth=2, label="Fused track (federated KF)")

    plt.plot(truths[:, 0], truths[:, 1], "k-o", markersize=4,
              linewidth=2, label="Ground truth")
    plt.scatter(truths[0, 0], truths[0, 1], s=80, color="black",
                marker="^", zorder=5, label="Start")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Federated KF vs. convex combination vs. ground truth")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
