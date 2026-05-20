# -*- coding: utf-8 -*-
"""Right-hand end-effector trajectory (XZ plane)."""

import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

csv_path = os.path.join(os.path.dirname(__file__), "experiment_data.csv")
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found."); sys.exit(1)

with open(csv_path, "r") as f:
    header = next(csv.reader(f))
    data = np.array([[float(v) for v in row] for row in csv.reader(f)])

col = {n: i for i, n in enumerate(header)}

# FK positions
fk_x = data[:, col["right_fk_x"]]
fk_z = data[:, col["right_fk_z"]]

# Target position
tgt_x = data[0, col["right_target_x"]]
tgt_z = data[0, col["right_target_z"]]

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fk_x, fk_z, "b-", linewidth=1.2, label="FK actual")
ax.scatter(fk_x[0], fk_z[0], color="blue", marker="o", s=60,
           zorder=5, label="start")
ax.scatter(fk_x[-1], fk_z[-1], color="red", marker="x", s=80,
           zorder=5, linewidths=2, label="end")
ax.scatter(tgt_x, tgt_z, color="green", marker="+", s=100,
           zorder=5, linewidths=2, label="target")

ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_title(f"Right hand trajectory (XZ plane) | target dz={tgt_z - fk_z[0]:.2f} m")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

out = os.path.join(os.path.dirname(__file__), "end_effector.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
