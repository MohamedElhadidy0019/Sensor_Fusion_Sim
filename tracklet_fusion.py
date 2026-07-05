"""Tracklet fusion via the information filter.

The Fusion Center keeps a single persistent central track (x, P), predicted
forward every step like an ordinary KF. Each sensor also keeps its own
persistent local track. Instead of sending its raw posterior to the FC, each
sensor reports the *information increment* its own update contributed this
step (relative to its own prior) -- and the FC adds that onto its own prior.

Reference: deck 5, p13-14 (eq. 926-932).
"""

import numpy as np


class TrackletFusion:
    """Central track + N persistent local sensor tracks, fused via the
    information filter.
    """

    def __init__(self, n_sensors, F, H, Q, R, x0, P0):
        self.n_sensors = n_sensors
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        # Fusion Center's own persistent central track.
        self.x = x0.copy()
        self.P = P0.copy()

        # One independent, persistent local track per sensor.
        self.x_local = [x0.copy() for _ in range(n_sensors)]
        self.P_local = [P0.copy() for _ in range(n_sensors)]

    def _information_filter_update(self, s, z):
        """Predict + update sensor s's own local track, and return its
        information increment (I^s, i^s) for this step, relative to its own
        prior:

            I^s = (P^s_k|k)^-1 - (P^s_k|k-1)^-1
            i^s = (P^s_k|k)^-1 x^s_k|k - (P^s_k|k-1)^-1 x^s_k|k-1
        """
        # Predict
        x_pred = self.F @ self.x_local[s]
        P_pred = self.F @ self.P_local[s] @ self.F.T + self.Q
        
        # Update
        nu = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        W = P_pred @ self.H.T @ np.linalg.inv(S)
        x_upd = x_pred + W @ nu
        P_upd = P_pred - W @ S @ W.T
        # note that P_upd is P^s_k|k, P_pred is P^s_k|k-1
        I = np.linalg.inv(P_upd) - np.linalg.inv(P_pred)
        
        # note that x_upd is x^s_k|k, x_pred is x^s_k|k-1
        i = np.linalg.inv(P_upd) @ x_upd - np.linalg.inv(P_pred) @ x_pred
        
        # Save the updated local track.
        self.x_local[s] = x_upd
        self.P_local[s] = P_upd 
        
        return I, i
                

    def _central_fusion(self, increments):
        """Predict the FC's own central track forward, then fuse in the S
        information increments from this step:

            (P_k|k)^-1       = (P_k|k-1)^-1         + sum_s I^s
            (P_k|k)^-1 x_k|k = (P_k|k-1)^-1 x_k|k-1 + sum_s i^s
        """
        # Central Fusion Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        i_pred = np.linalg.inv(P_pred) @ x_pred
        I_pred = np.linalg.inv(P_pred)
        
        # Update with Tracklet Fusion method
        i_upd = i_pred + sum(i for I, i in increments)
        I_upd = I_pred + sum(I for I, i in increments)
        
        P_upd = np.linalg.inv(I_upd)
        x_upd = P_upd @ i_upd
        
        self.x = x_upd
        self.P = P_upd
        
        
        

    def step(self, measurements):
        """Advance one step: update each sensor's local track (information
        filter), then fuse the increments into the FC's central track.

        measurements: list of S position measurements z_1..z_S.
        Returns (x_fused, P_fused).
        """
        increments = [self._information_filter_update(s, z)
                      for s, z in enumerate(measurements)]
        self._central_fusion(increments)
        
        return self.x, self.P


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from convex_comb import ConvexCombinationFusion
    from sim import Sim, cv_model

    N = 50
    n_sensors = 100

    sim = Sim(n_sensors=n_sensors, seed=0)
    F, H, Q, R = cv_model()

    # Uninformative prior: filters don't get to "know" the true start state.
    x0 = np.zeros(4)
    P0 = 1000.0 * np.eye(4)

    tracklet = TrackletFusion(n_sensors, F, H, Q, R, x0, P0)
    convex = ConvexCombinationFusion(n_sensors, F, H, Q, R, x0, P0)

    truths = []
    tracklet_track = []
    convex_track = []

    for _ in range(N):
        truth, measurements = sim.step()

        # Both methods see the exact same measurements, for a fair comparison.
        x_tracklet, _ = tracklet.step(measurements)
        x_convex, _ = convex.step(measurements)

        truths.append(truth)
        tracklet_track.append(x_tracklet)
        convex_track.append(x_convex)

    truths = np.array(truths)                    # (N, 4)
    tracklet_track = np.array(tracklet_track)     # (N, 4)
    convex_track = np.array(convex_track)         # (N, 4)

    # Position RMSE over the whole run: sqrt(mean over time of squared
    # Euclidean distance between estimate and ground truth).
    rmse_tracklet = np.sqrt(np.mean(np.sum((tracklet_track[:, :2] - truths[:, :2])**2, axis=1)))
    rmse_convex = np.sqrt(np.mean(np.sum((convex_track[:, :2] - truths[:, :2])**2, axis=1)))
    print(f"RMSE (tracklet fusion):      {rmse_tracklet:.3f}")
    print(f"RMSE (convex combination):   {rmse_convex:.3f}")

    plt.figure(figsize=(8, 8))

    plt.plot(convex_track[:, 0], convex_track[:, 1], "c-o", markersize=4,
              linewidth=2, label="Fused track (convex combination)")
    plt.plot(tracklet_track[:, 0], tracklet_track[:, 1], "m-o", markersize=4,
              linewidth=2, label="Fused track (tracklet fusion)")

    plt.plot(truths[:, 0], truths[:, 1], "k-o", markersize=4,
              linewidth=2, label="Ground truth")
    plt.scatter(truths[0, 0], truths[0, 1], s=80, color="black",
                marker="^", zorder=5, label="Start")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Tracklet fusion vs. convex combination vs. ground truth")
    plt.legend(loc="upper left")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
