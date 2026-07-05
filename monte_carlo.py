"""Monte Carlo comparison of all four fusion methods.

Runs the simulation n_runs times (a different random seed each time), and
for each run computes every method's position RMSE against ground truth.
Reports mean +- std RMSE per method across all runs -- a much more robust
comparison than a single run, since it washes out single-run noise luck.
"""

import csv

import numpy as np
import matplotlib.pyplot as plt

from sim import Sim, cv_model
from convex_comb import ConvexCombinationFusion
from tracklet_fusion import TrackletFusion
from federated_filter import FederatedFilter
from distributed_filter import DistributedFilter

N = 50
n_sensors = 4
n_runs = 100

F, H, Q, R = cv_model()
x0 = np.zeros(4)
P0 = 1000.0 * np.eye(4)

method_classes = {
    "Convex combination": ConvexCombinationFusion,
    "Tracklet fusion":     TrackletFusion,
    "Federated KF":        FederatedFilter,
    "Distributed KF":      DistributedFilter,
}

rmses = {name: [] for name in method_classes}

for run in range(n_runs):
    sim = Sim(n_sensors=n_sensors, seed=run)
    methods = {name: cls(n_sensors, F, H, Q, R, x0, P0)
               for name, cls in method_classes.items()}

    truths = []
    tracks = {name: [] for name in methods}

    for _ in range(N):
        truth, measurements = sim.step()
        truths.append(truth)

        # Every method sees the exact same measurements, for a fair comparison.
        for name, fusion in methods.items():
            x_fused, _ = fusion.step(measurements)
            tracks[name].append(x_fused)

    truths = np.array(truths)
    for name, track in tracks.items():
        track = np.array(track)
        rmse = np.sqrt(np.mean(np.sum((track[:, :2] - truths[:, :2])**2, axis=1)))
        rmses[name].append(rmse)

print(f"Monte Carlo RMSE over {n_runs} runs (N={N} steps, S={n_sensors} sensors):")
print(f"  {'Method':<20s} {'mean':>8s}   {'std':>8s}")
summary = []
for name, values in rmses.items():
    values = np.array(values)
    mean, std = values.mean(), values.std()
    print(f"  {name:<20s} {mean:8.3f} +/- {std:.3f}")
    summary.append((name, mean, std))

with open("monte_carlo_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "rmse_mean", "rmse_std"])
    writer.writerows(summary)
print("\nSaved table to monte_carlo_results.csv")

plt.figure(figsize=(8, 5))
names = [s[0] for s in summary]
means = [s[1] for s in summary]
stds = [s[2] for s in summary]
plt.bar(names, means, yerr=stds, capsize=6, color="tab:blue", alpha=0.8)
plt.ylabel("Position RMSE [m]")
plt.title(f"Mean +/- std RMSE over {n_runs} runs (S={n_sensors} sensors, N={N} steps)")
plt.tight_layout()
plt.savefig("monte_carlo_comparison.png", dpi=150)
print("Saved plot to monte_carlo_comparison.png")
plt.show()
