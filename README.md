# Unitree H1 机器人 —— 运动学与动力学分析

基于 MuJoCo 物理引擎，对 Unitree H1 人形机器人的上身（躯干 + 双臂）进行运动学与动力学建模、求解与验证。

**简介**

本工程包含一个完整的演示：使用逆运动学（IK）求解右手上举目标位姿，生成关节空间 S 曲线轨迹，执行逆动力学（ID）计算所需关节力矩，并将力矩输入正动力学（FD）验证加速度一致性。运行结果会记录为 CSV 数据，配套脚本用于绘图与分析。

**目录说明**

- `reach_upward.py`：主实验脚本。执行 IK → 轨迹规划 → ID/FD → 可视化，并生成 `experiment_data.csv`。
- `plot_results.py`：读取 `experiment_data.csv` 并生成综合分析图 `experiment_results.png`。
- `plot_trajectory.py`, `plot_torques.py`, `plot_fd_error.py`, `plot_end_effector.py`：辅助绘图脚本，分别用于轨迹、力矩、FD 误差与末端示意。
- `unitree_h1/`：包含 H1 机器人的 MJCF 模型文件（`scene.xml`、`h1.xml`）与资源。
<!-- - `实验报告.md`、`源代码分析.md`：实验报告与源码分析说明文档。 -->

**运行环境与依赖**

- 操作系统：Windows / Linux / macOS 均可（示例在 Windows 下测试）。
- Python：建议 3.8+
- 必需 Python 包（至少安装下列包）：
	- `numpy`, `matplotlib`, `mink`（逆运动学库）, `mujoco`（MuJoCo Python 接口）
	- 其它脚本中使用的库请参考源文件顶部的 `import` 语句

安装示例（虚拟环境中执行）：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy matplotlib
# 请根据 MuJoCo 与 mink 的安装说明安装对应包与运行时（通常需先安装 MuJoCo 本体）
```

**快速开始**

1. 确保 MuJoCo 已安装并可由 Python 的 `mujoco` 模块访问。
2. 在项目根目录运行主实验：

```bash
python reach_upward.py
```

运行结束后会在同目录生成 `experiment_data.csv`。

3. 生成分析图表：

```bash
python plot_results.py
```

输出：`experiment_results.png`（以及其它由单独 plot 脚本生成的图片）。

**注意**

- `reach_upward.py` 中使用了 MuJoCo 与 `mink`（IK）库，运行前请先完成其安装与许可配置。
- 若在 Windows 控制台出现中文字体显示问题，可使用支持中文的终端或调整 matplotlib 字体设置。
