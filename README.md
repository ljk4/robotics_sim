# H1 固定骨盆双臂系统 — 运动学与多体动力学分析

基于 MuJoCo 物理引擎和 Unitree H1 人形机器人模型，对固定骨盆双臂系统（9 自由度：躯干 1 + 左臂 4 + 右臂 4）进行系统的运动学与多体动力学分析。共包含六个实验，覆盖 FK、IK、ID、FD 以及质量矩阵耦合分析。

> 原始 H1 模型已移除下半身（双腿 + 骨盆自由关节），骨盆直接固定于世界坐标系。仅保留上半身 9 个关节。

## 环境搭建

需要 conda 环境，依赖以下 Python 包：

| 包 | 用途 |
|------|------|
| `mujoco` | 物理引擎 |
| `mink` | 微分 IK + QP 求解 |
| `numpy` | 数值计算 |
| `matplotlib` | 绘图 |
| `opencv-python` | 视频录制（可选，缺失时自动跳过） |

创建环境：

```bash
conda create -n robotics python=3.11
conda activate robotics
pip install mujoco mink numpy matplotlib opencv-python
```

## 运行

```bash
conda activate robotics
python dual_arm_analysis.py
```

所有输出文件（图片 + 视频 + 统计 CSV）保存到 `results/` 目录，详细实验日志保存到 `results/experiment_log.txt`。

## 实验内容

| 实验 | 功能 | 关键输出 |
|------|------|---------|
| 1 FK 工作空间 | 5000 组随机关节角 → 3D 手腕点云 | `fig2_fk_workspace.png` |
| 2 双臂 IK | 对称 FK 采样 → IK 追踪，验证收敛 | `fig3_ik_convergence.png`, `fig3b_ik_targets.png`, `exp2_ik_pose.png` |
| 3 双臂提升 (ID) | S 曲线轨迹 + 逆动力学力矩 | `fig4_dual_lift_torques.png`, `fig5_torso_torque.png`, `exp3_dual_lift.mp4` |
| 4 动态耦合 | 单臂 vs 双臂提升，耦合效应 | `fig6_coupling_torso.png`, `fig7_coupling_shoulders.png` |
| 5 质量矩阵 | 9×9 质量矩阵热力图 | `fig8_mass_matrix.png` |
| 6 ID-FD 验证 | 正/逆动力学闭环自洽性验证 | `fig9_fd_error.png`, `fig10_fd_stats.png`, `exp6_fd_stats.csv` |

## 关键结果

| 指标 | 数值 |
|------|------|
| FK 工作空间 X/Y/Z 范围 | ±0.80 m / ±0.80 m / 0.91–2.11 m |
| IK 收敛步数 | **7 步**到亚毫米精度（左 0.46 mm，右 0.34 mm） |
| 双臂上举肩关节峰值力矩 | 1.88 Nm |
| 单/双臂躯干力矩耦合差异 | 均值 0.0632 Nm，峰值 0.1288 Nm |
| 质量矩阵躯干自惯量 | 0.5922 kg·m² |
| ID-FD 闭环误差 (MAE) | 2.38×10⁻¹⁶ rad/s²（机器精度） |

## 文件结构

```
dual_arm_analysis.py          # 主脚本
unitree_h1/                   # MuJoCo 模型文件
  ├── h1.xml                  # 机器人模型（已移除下半身）
  ├── scene.xml               # 场景配置
  └── assets/                 # 网格文件
results/                      # 输出结果
  ├── fig*.png                # 11 张实验图表
  ├── exp*_stats.csv          # 数值统计
  ├── exp3_dual_lift.mp4      # 运动视频
  ├── exp2_ik_pose.png        # IK 最终姿态
  └── experiment_log.txt      # 详细实验日志
```
