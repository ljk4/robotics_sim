# -*- coding: utf-8 -*-
"""
H1 upper-body kinematics & dynamics experiment -- result visualization
======================================================================
Reads experiment_data.csv, generates 6-panel analysis figure,
saves as experiment_results.png
"""

import csv
import os
import sys

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ---- Chinese font setup (for chart labels) ----
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 读取数据 ----
csv_path = os.path.join(os.path.dirname(__file__), "experiment_data.csv")
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Run upper_body_experiment.py first.")
    sys.exit(1)

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(v) for v in row] for row in reader], dtype=np.float64)

col_idx = {name: i for i, name in enumerate(header)}
t = data[:, col_idx["t"]]

upper_joints = [
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll",
    "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow",
]
joint_labels = [
    "躯干 (torso)",
    "左肩俯仰", "左肩滚转", "左肩偏航", "左肘",
    "右肩俯仰", "右肩滚转", "右肩偏航", "右肘",
]
n_joints = len(upper_joints)

# 提取数据
def col_vec(col_name):
    return data[:, col_idx[col_name]]

q_data   = np.column_stack([col_vec(f"q_{j}")   for j in upper_joints])
qd_data  = np.column_stack([col_vec(f"qd_{j}")  for j in upper_joints])
tau_data = np.column_stack([col_vec(f"tau_{j}") for j in upper_joints])
qacc_err_data = np.column_stack([col_vec(f"qacc_err_{j}") for j in upper_joints])

left_target  = np.column_stack([col_vec(f"left_target_{a}")  for a in "xyz"])
right_target = np.column_stack([col_vec(f"right_target_{a}") for a in "xyz"])
left_fk      = np.column_stack([col_vec(f"left_fk_{a}")      for a in "xyz"])
right_fk     = np.column_stack([col_vec(f"right_fk_{a}")     for a in "xyz"])
left_err  = col_vec("left_err") * 1000   # → mm
right_err = col_vec("right_err") * 1000  # → mm

# ---- 创建图表 ----
fig = plt.figure(figsize=(22, 26))
gs_main = GridSpec(6, 1, figure=fig, hspace=0.4)

colors = plt.cm.tab10(np.linspace(0, 1, n_joints))


def make_row(subplot_spec, ydata, ylabel, ylimits=None):
    """在 subplot_spec 内创建 1×9 子图行。"""
    gs_row = GridSpecFromSubplotSpec(1, n_joints, subplot_spec=subplot_spec)
    axes = []
    for i in range(n_joints):
        ax = fig.add_subplot(gs_row[0, i])
        ax.plot(t, ydata[:, i], color=colors[i], linewidth=0.6)
        ax.set_title(joint_labels[i], fontsize=8, pad=2)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if ylimits is not None:
            margin = 0.05 * (ylimits[1] - ylimits[0])
            ax.set_ylim(ylimits[0] - margin, ylimits[1] + margin)
        axes.append(ax)
    return axes


# ===== 第 1 行: 关节角度 =====
make_row(gs_main[0], q_data, "rad")
fig.text(0.02, 0.935, "关节角度 q(t)", fontsize=11, fontweight="bold", va="center",
         transform=fig.transFigure)

# ===== 第 2 行: 关节速度 =====
make_row(gs_main[1], qd_data, "rad/s")
fig.text(0.02, 0.782, "关节速度 q̇(t)", fontsize=11, fontweight="bold", va="center",
         transform=fig.transFigure)

# ===== 第 3 行: 关节力矩 (ID) =====
make_row(gs_main[2], tau_data, "N·m")
# 为每个子图添加零线
for ax in fig.axes[-n_joints:]:
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
fig.text(0.02, 0.629, "逆动力学关节力矩 τ(t)", fontsize=11, fontweight="bold",
         va="center", transform=fig.transFigure)

# ===== 第 4 行: 末端轨迹对比 =====
ax_traj = fig.add_subplot(gs_main[3])
ax_traj.plot(left_target[:, 0], left_target[:, 2], "--", color="blue",
             alpha=0.5, linewidth=1, label="左手目标")
ax_traj.plot(right_target[:, 0], right_target[:, 2], "--", color="green",
             alpha=0.5, linewidth=1, label="右手目标")
ax_traj.plot(left_fk[:, 0], left_fk[:, 2], color="blue",
             linewidth=1.2, label="左手 FK 实际")
ax_traj.plot(right_fk[:, 0], right_fk[:, 2], color="green",
             linewidth=1.2, label="右手 FK 实际")
ax_traj.scatter(*left_fk[0, [0, 2]], color="blue", marker="o", s=40, zorder=5)
ax_traj.scatter(*right_fk[0, [0, 2]], color="green", marker="o", s=40, zorder=5)
ax_traj.set_xlabel("X (m)", fontsize=9)
ax_traj.set_ylabel("Z (m)", fontsize=9)
ax_traj.set_title("末端轨迹 (XZ 平面) — 目标圆 vs FK 实际", fontsize=10, fontweight="bold")
ax_traj.legend(fontsize=7, loc="upper right")
ax_traj.set_aspect("equal")
ax_traj.grid(True, alpha=0.3)

# ===== 第 5 行: 位置误差 =====
ax_err = fig.add_subplot(gs_main[4])
ax_err.plot(t, left_err, color="blue", linewidth=0.8,
            label=f"左手 (均值={np.mean(left_err):.2f} mm)")
ax_err.plot(t, right_err, color="green", linewidth=0.8,
            label=f"右手 (均值={np.mean(right_err):.2f} mm)")
ax_err.set_xlabel("时间 (s)", fontsize=9)
ax_err.set_ylabel("位置误差 (mm)", fontsize=9)
ax_err.set_title("末端位置误差 (FK 实际 vs IK 目标)", fontsize=10, fontweight="bold")
ax_err.legend(fontsize=8)
ax_err.grid(True, alpha=0.3)

# ===== 第 6 行: FD 加速度误差 =====
make_row(gs_main[5], qacc_err_data, "rad/s²")
for ax in fig.axes[-n_joints:]:
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
fig.text(0.02, 0.062, "FD 加速度误差 (q̈_FD − q̈_approx)", fontsize=11,
         fontweight="bold", va="center", transform=fig.transFigure)

# ---- 保存 ----
out_path = os.path.join(os.path.dirname(__file__), "experiment_results.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Figure saved to: {out_path}")
