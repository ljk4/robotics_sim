# -*- coding: utf-8 -*-
"""Joint position / velocity / acceleration vs time (9-DOF upper body)."""

import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

csv_path = os.path.join(os.path.dirname(__file__), "experiment_data.csv")
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found."); sys.exit(1)

with open(csv_path, "r") as f:
    header = next(csv.reader(f))
    data = np.array([[float(v) for v in row] for row in csv.reader(f)])

col = {n: i for i, n in enumerate(header)}
t = data[:, col["t"]]

upper_joints = ["torso","left_shoulder_pitch","left_shoulder_roll",
    "left_shoulder_yaw","left_elbow","right_shoulder_pitch",
    "right_shoulder_roll","right_shoulder_yaw","right_elbow"]
labels = ["torso","L-shld-pitch","L-shld-roll","L-shld-yaw","L-elbow",
          "R-shld-pitch","R-shld-roll","R-shld-yaw","R-elbow"]
n = len(upper_joints)
colors = plt.cm.tab10(np.linspace(0, 1, n))

fig = plt.figure(figsize=(18, 10))
gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=fig.add_gridspec(3,1)[:3], hspace=0.35)

for row_idx, (prefix, ylabel, title) in enumerate([
    ("q",   "rad",     "Joint position q(t)"),
    ("qd",  "rad/s",   "Joint velocity qd(t)"),
    ("qdd", "rad/s^2", "Joint acceleration qdd(t)"),
]):
    gs_row = GridSpecFromSubplotSpec(1, n, subplot_spec=gs[row_idx])
    for i in range(n):
        ax = fig.add_subplot(gs_row[0, i])
        ax.plot(t, data[:, col[f"{prefix}_{upper_joints[i]}"]],
                color=colors[i], linewidth=0.8)
        if i == 0: ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(labels[i], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

out = os.path.join(os.path.dirname(__file__), "trajectory.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
