"""Run all four fusion methods on one shared measurement stream and compare
them against ground truth and each other.

Just imports and wires together what's already implemented in
convex_comb.py, tracklet_fusion.py, federated_filter.py and
distributed_filter.py -- no new fusion logic here.
"""

import numpy as np
import matplotlib.pyplot as plt

from sim import Sim, cv_model
from convex_comb import ConvexCombinationFusion
from tracklet_fusion import TrackletFusion
from federated_filter import FederatedFilter
from distributed_filter import DistributedFilter

N = 50
n_sensors = 4

sim = Sim(n_sensors=n_sensors, seed=0)
F, H, Q, R = cv_model()

# Uninformative prior: filters don't get to "know" the true start state.
x0 = np.zeros(4)
P0 = 1000.0 * np.eye(4)

methods = {
    "Convex combination": ConvexCombinationFusion(n_sensors, F, H, Q, R, x0, P0),
    "Tracklet fusion":     TrackletFusion(n_sensors, F, H, Q, R, x0, P0),
    "Federated KF":        FederatedFilter(n_sensors, F, H, Q, R, x0, P0),
    "Distributed KF":      DistributedFilter(n_sensors, F, H, Q, R, x0, P0),
}

truths = []
tracks = {name: [] for name in methods}

for _ in range(N):
    truth, measurements = sim.step()
    truths.append(truth)

    # Every method sees the exact same measurements, for a fair comparison.
    for name, fusion in methods.items():
        x_fused, _ = fusion.step(measurements)
        tracks[name].append(x_fused)

truths = np.array(truths)  # (N, 4)
for name in tracks:
    tracks[name] = np.array(tracks[name])  # (N, 4)

# Position RMSE over the whole run: sqrt(mean over time of squared
# Euclidean distance between estimate and ground truth).
print(f"Position RMSE over the run (N={N} steps, S={n_sensors} sensors):")
for name, track in tracks.items():
    rmse = np.sqrt(np.mean(np.sum((track[:, :2] - truths[:, :2])**2, axis=1)))
    print(f"  {name:<20s} {rmse:.3f}")

plt.figure(figsize=(8, 8))

colors = {"Convex combination": "tab:cyan", "Tracklet fusion": "tab:pink",
          "Federated KF": "tab:orange", "Distributed KF": "tab:green"}
for name, track in tracks.items():
    plt.plot(track[:, 0], track[:, 1], "-o", markersize=4, linewidth=2,
              color=colors[name], label=f"Fused track ({name})")

plt.plot(truths[:, 0], truths[:, 1], "k-o", markersize=4, linewidth=2,
          label="Ground truth")
plt.scatter(truths[0, 0], truths[0, 1], s=80, color="black", marker="^",
            zorder=5, label="Start")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title(f"All four fusion methods vs. ground truth (S={n_sensors}, N={N})")
plt.legend(loc="upper left")
plt.axis("equal")
plt.tight_layout()
plt.savefig("run_comparison.png", dpi=150)
print("\nSaved plot to run_comparison.png")
plt.show()
