# -*- coding: utf-8 -*-
"""
H1 人形机器人 —— 右手竖直上举实验（带可视化）
======================================================
本脚本演示一个完整的 "运动学 + 动力学" 实验流程：

  第1步 (IK):   已知右手目标位置（向上 15cm），用逆运动学求出关节角度
  第2步 (轨迹): 用 S 曲线（五次多项式）在关节空间中生成平滑运动轨迹
  第3步 (ID):   已知运动（角度/速度/加速度），用逆动力学反算关节力矩
  第4步 (FD):   将力矩输入正动力学，验证加速度是否一致（ID→FD 闭环）

最终数据保存为 experiment_data.csv，可用 plot_*.py 生成图表。

运行方式：
    python reach_upward.py
"""

# =============================================================================
# 第0部分：导入 Python 库
# =============================================================================

# csv:     用于写入 experiment_data.csv 数据文件
# os:      用于拼接文件路径
# sys:     系统相关功能
import csv, os, sys

# mujoco:  MuJoCo 物理引擎的 Python 接口
#   - MjModel:     机器人模型（包含关节、刚体、几何体等静态信息）
#   - MjData:      仿真数据（包含关节角度、速度、加速度等动态状态）
#   - viewer:      3D 可视化窗口
import mujoco, mujoco.viewer

# numpy: 数值计算库，提供数组、矩阵运算
#   - np.array: 创建数组
#   - np.linalg.norm: 计算向量长度（欧氏距离）
#   - np.block: 拼接矩阵
import numpy as np

# mink: 逆运动学库（基于 MuJoCo）
#   - Configuration:  包装 MuJoCo 模型和数据，管理机器人状态
#   - solve_ik:       求解逆运动学（核心函数）
#   - FrameTask:      一个 IK 任务——让某个坐标系（body/site）跟踪目标位姿
#   - PostureTask:    正则化任务——让关节尽量保持参考姿态
#   - ComTask:        质心任务——让质心保持在目标位置
#   - CollisionAvoidanceLimit: 碰撞避免约束（硬约束）
#   - ConfigurationLimit:      关节限位约束（硬约束）
#   - SE3:            刚体变换（位置 + 姿态），属于李群 SE(3)
import mink
from mink import CollisionAvoidanceLimit, ConfigurationLimit

# RateLimiter: 控制仿真循环频率（确保每步时间精确）
from loop_rate_limiters import RateLimiter


# =============================================================================
# 实验参数（可根据需要修改）
# =============================================================================

TRAJ_DURATION = 2.0       # 轨迹运动时长（秒）
FREQ = 200                # 控制频率（Hz）—— 每秒执行 200 个控制步
DT = 1.0 / FREQ           # 每步的时间间隔 = 0.005 秒
TARGET_DZ = 0.15          # 右手上举的高度（米），这里是 15cm
IK_STEPS = 400            # IK 求解的最大迭代步数


# =============================================================================
# S 曲线轨迹规划（五次多项式平滑阶跃）
# =============================================================================
# 为什么要用 S 曲线？
#   如果让关节直接从起点角度跳到终点角度，速度和加速度会瞬间突变（冲击），
#   不仅不真实，还会导致动力学计算出现无穷大的力矩。
#
#   S 曲线（五次多项式 smooth step）保证：
#     - 位置从起点连续过渡到终点
#     - 起点和终点的速度 = 0
#     - 起点和终点的加速度 = 0
#   这样运动从头到尾都是平滑的。
#
# 数学原理：
#   s(u) = 10u³ - 15u⁴ + 6u⁵   (u ∈ [0,1])
#
#   验证端点条件：
#     s(0)=0,  s(1)=1          ← 位置从 0 到 1
#     s'(0)=0, s'(1)=0         ← 起点终点速度为零
#     s''(0)=0, s''(1)=0       ← 起点终点加速度为零
#
#   导数（用于计算速度和加速度）：
#     s'(u)  = 30u² - 60u³ + 30u⁴
#     s''(u) = 60u - 180u² + 120u³
#
#   映射到实际时间 t ∈ [0, T]：
#     q(t)   = q₀ + (q_f - q₀) · s(t/T)              ← 位置
#     q̇(t)   = (q_f - q₀) · s'(t/T) / T              ← 速度（链式法则）
#     q̈(t)   = (q_f - q₀) · s''(t/T) / T²            ← 加速度

def s_curve_sample(t, T, q0, qf):
    """
    在时刻 t 采样 S 曲线轨迹。

    参数:
        t  (float):     当前时间（秒）
        T  (float):     轨迹总时长（秒）
        q0 (ndarray):   起点关节角度（19 个铰链关节）
        qf (ndarray):   终点关节角度

    返回:
        q   (ndarray):  当前关节角度
        qd  (ndarray):  当前关节速度（角速度, rad/s）
        qdd (ndarray):  当前关节加速度（角加速度, rad/s²）
    """
    # u = 归一化时间，范围 [0, 1]
    # max/min 防止浮点误差导致 u 超出 [0,1]
    u = max(0.0, min(1.0, t / T))

    # 五次多项式 s(u) 及其导数
    s   = 10*u**3 - 15*u**4 + 6*u**5          # 位置曲线
    sd  = 30*u**2 - 60*u**3 + 30*u**4          # s'(u) = 一阶导数
    sdd = 60*u   - 180*u**2 + 120*u**3         # s''(u) = 二阶导数

    # 映射到关节空间
    q   = q0 + (qf - q0) * s                   # 角度：从 q0 平滑过渡到 qf
    qd  = (qf - q0) * (sd / T)                 # 角速度：链式法则 dq/dt = dq/du · du/dt
    qdd = (qf - q0) * (sdd / (T * T))          # 角加速度：d²q/dt² = d²q/du² · (du/dt)²

    return q, qd, qdd


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 第1部分：加载机器人模型
    # -------------------------------------------------------------------------

    # from_xml_path: 从 MJCF（MuJoCo XML）文件加载机器人模型
    # scene.xml 包含 H1 机器人 + 地面 + 灯光 + 相机 + 关键帧
    model = mujoco.MjModel.from_xml_path("unitree_h1/scene.xml")

    # Configuration: mink 包装器，把 model 和 data 绑定在一起
    # 它会自动管理关节位置、调用正向运动学等
    configuration = mink.Configuration(model)

    # data: 指向 configuration 内部的 MjData 对象
    # MjData 存储仿真运行时状态：qpos（关节位置）、qvel（速度）、
    #   site_xpos（site 世界坐标）、qacc（加速度）等
    # 注意：必须用 configuration.data，不能另外创建 MjData(model)
    data = configuration.data

    # -------------------------------------------------------------------------
    # 第2部分：定义 IK 任务
    # -------------------------------------------------------------------------
    # IK（逆运动学）需要同时满足多个任务。每个任务有一个"代价权重"，
    # 权重越大表示优先级越高。mink 会求解一个最小化加权误差的二次规划（QP）。
    #
    # 本实验的任务层次（从高到低优先级）：
    #   200: 双脚保持不动、右手到达目标、左手保持不动、质心保持
    #   10:  骨盆保持直立姿态
    #   1:   所有关节尽量接近初始站立姿态（正则化）

    feet  = ["right_foot", "left_foot"]       # 双脚 site 名称
    hands = ["right_wrist", "left_wrist"]     # 双手 site 名称（site 是模型上标记的参考点）

    # 上半身 9 个关节（本实验的分析对象）
    upper_joint_names = [
        "torso",                               # 躯干旋转（1 个关节）
        "left_shoulder_pitch", "left_shoulder_roll",
        "left_shoulder_yaw", "left_elbow",     # 左臂（4 个关节：肩 3 + 肘 1）
        "right_shoulder_pitch", "right_shoulder_roll",
        "right_shoulder_yaw", "right_elbow",   # 右臂（4 个关节）
    ]

    # 获取每个关节在 qpos（位置数组）和 qvel（速度数组）中的索引
    # jnt_qposadr: 该关节在 qpos 数组中的起始索引
    # jnt_dofadr:  该关节在 qvel 数组中的起始索引（每个铰链关节占 1 个自由度）
    upper_qpos_ids = [model.jnt_qposadr[model.joint(j).id] for j in upper_joint_names]
    upper_dof_ids  = [model.jnt_dofadr[model.joint(j).id]  for j in upper_joint_names]

    # ---- 定义 IK 任务列表 ----

    # 任务1: 骨盆姿态保持（权重：方向=10, 位置=0 表示不关心骨盆平移）
    pelvis_orientation_task = mink.FrameTask(
        frame_name="pelvis",    # 骨盆 body 名称
        frame_type="body",      # frame 类型为 body
        position_cost=0.0,      # 不关心位置
        orientation_cost=10.0,  # 姿态权重=10
    )

    # 任务2: 关节姿态正则化（权重=1, 让关节尽量靠近参考姿态）
    posture_task = mink.PostureTask(model, cost=1.0)

    # 任务3: 质心保持（权重=200, 最高优先级, 防止机器人倾斜/摔倒）
    com_task = mink.ComTask(cost=200.0)

    # 组装任务列表（Python 3.8+ 海象运算符 := 同时赋值和添加到列表）
    tasks = [pelvis_orientation_task, posture_task, com_task]

    # 任务4-5: 双脚固定（权重=200 位置 + 10 姿态, 表示双脚不能移动）
    # lm_damping: Levenberg-Marquardt 阻尼, 防止关节在奇异构型附近产生过大速度
    feet_tasks = []
    for foot in feet:
        tsk = mink.FrameTask(
            frame_name=foot, frame_type="site",
            position_cost=200.0, orientation_cost=10.0, lm_damping=1.0,
        )
        feet_tasks.append(tsk)
        tasks.append(tsk)

    # 任务6: 右手位置跟踪（权重=200, 只关心位置不关心姿态）
    # 这是本实验的主要任务——驱动右手到达目标位置
    right_hand_task = mink.FrameTask(
        frame_name="right_wrist", frame_type="site",
        position_cost=200.0, orientation_cost=0.0, lm_damping=1.0,
    )

    # 任务7: 左手位置保持（权重=200, 左手停在原处）
    left_hand_task = mink.FrameTask(
        frame_name="left_wrist", frame_type="site",
        position_cost=200.0, orientation_cost=0.0, lm_damping=1.0,
    )
    tasks.extend([right_hand_task, left_hand_task])

    # -------------------------------------------------------------------------
    # 第3部分：碰撞避免约束（硬约束）
    # -------------------------------------------------------------------------
    # 与软任务（加权最小化）不同，碰撞避免是硬约束——
    # 求解器必须保证几何体之间保持最小距离，否则 QP 无解。
    #
    # 这里保护：
    #   左臂（上臂、前臂、肘球）不碰到躯干/头部/头盔
    #   右臂同理
    #
    # minimum_distance_from_collisions=0.05 表示保持至少 5 厘米间距

    collision_limit = CollisionAvoidanceLimit(
        model,
        geom_pairs=[
            # 左臂 vs 躯干/头部
            (["left_upper_arm","left_forearm","left_elbow_sphere"],
             ["torso","head","helmet"]),
            # 右臂 vs 躯干/头部
            (["right_upper_arm","right_forearm","right_elbow_sphere"],
             ["torso","head","helmet"]),
        ],
        minimum_distance_from_collisions=0.05,
    )

    # 把约束打包传入 solve_ik（注意：一旦提供 limits，ConfigurationLimit 必须显式添加）
    limits = [ConfigurationLimit(model), collision_limit]

    # -------------------------------------------------------------------------
    # 第4部分：初始化——加载站立姿态
    # -------------------------------------------------------------------------

    # 从 scene.xml 中名为 "stand" 的关键帧恢复机器人状态
    # 关键帧包含一组预定义的关节角度（qpos），使机器人呈站立姿态
    configuration.update_from_keyframe("stand")

    # 将当前姿态设为 posture_task 的参考姿态（即"尽量保持这个姿势"）
    posture_task.set_target_from_configuration(configuration)

    # 将当前骨盆姿态设为 pelvis_orientation_task 的目标
    pelvis_orientation_task.set_target_from_configuration(configuration)

    # 调用正向运动学，更新 site_xpos（site 世界坐标）等派生数据
    # update_from_keyframe 只设置了 qpos，需要 mj_forward 才能计算出
    #   各刚体的世界位置、site 的世界坐标等
    mujoco.mj_forward(model, data)

    # 将当前质心位置设为 com_task 的目标（即"保持在当前位置"）
    # subtree_com[1] 是躯干子树（包含上半身和双臂）的质心位置
    com_task.set_target(data.subtree_com[1])

    # 将双脚当前位姿（位置 + 旋转矩阵）设为 foot_task 的目标
    for foot_task, foot in zip(feet_tasks, feet):
        sid = model.site(foot).id              # site 的索引编号
        fp = data.site_xpos[sid]               # site 世界坐标 (3,)
        fr = data.site_xmat[sid].reshape(3,3)  # site 旋转矩阵 (3,3)
        # 构造 4×4 齐次变换矩阵，用 SE3 包装
        foot_task.set_target(mink.SE3.from_matrix(
            np.block([[fr, fp[:, None]], [0,0,0,1]])
        ))

    # 记录左手当前位置，设为 left_hand_task 的目标（左手不动）
    left_sid = model.site("left_wrist").id
    left_hand_task.set_target(
        mink.SE3.from_translation(data.site_xpos[left_sid].copy())
    )

    # -------------------------------------------------------------------------
    # 第5部分：逆运动学求解（IK）
    # -------------------------------------------------------------------------
    # 目标：找到一组关节角度，使右手到达 target_pos（初始位置上方 15cm）
    #
    # 方法：迭代求解微分 IK
    #   每一步：
    #     1. 设置右手目标
    #     2. solve_ik() 求解 QP，得到关节速度 vel
    #     3. integrate_inplace() 积分速度 → 更新关节角度
    #     4. mj_forward() 更新 FK
    #     5. 检查右手是否到达目标（误差 < 1mm）
    #   重复直到收敛或达到最大步数

    right_sid = model.site("right_wrist").id
    right_init = data.site_xpos[right_sid].copy()    # 右手初始世界坐标

    # 目标位置 = 初始位置 + 竖直向上 TARGET_DZ 米
    target_pos = right_init + np.array([0.0, 0.0, TARGET_DZ])

    # 保存起点关节角度（用于 S 曲线轨迹）
    q0 = data.qpos.copy()

    print("Solving IK for final configuration ...")
    solver = "daqp"   # daqp = Dual Active-set QP solver

    for step in range(IK_STEPS):
        # 设置右手的 IK 目标（只关心位置，使用 from_translation 忽略姿态）
        right_hand_task.set_target(mink.SE3.from_translation(target_pos))

        # 核心：调用 solve_ik 求解关节速度
        #   参数:
        #     configuration: 当前机器人状态
        #     tasks:         IK 任务列表（7 个任务）
        #     dt:            时间步长
        #     solver:        QP 求解器名称
        #     damping:       阻尼系数（LM 阻尼，防止奇异构型）
        #     limits:        硬约束列表（关节限位 + 碰撞避免）
        #   返回:
        #     vel: 最优关节速度向量，维度 = nv（速度自由度总数）
        vel = mink.solve_ik(
            configuration, tasks, DT, solver, damping=1e-1, limits=limits
        )

        # 积分速度 → 更新关节角度
        # 这步会直接修改 data.qpos，并调用 mj_forward 更新 FK 派生数据
        configuration.integrate_inplace(vel, DT)

        # 再跑一次 FK（integrate_inplace 内部已调用, 这里为明确起见再调用一次）
        mujoco.mj_forward(model, data)

        # 检查右手当前 FK 位置与目标的距离
        err = np.linalg.norm(data.site_xpos[right_sid] - target_pos)

        # 如果位置误差小于 1 毫米，认为收敛，退出循环
        if err < 0.001:
            print(f"  Converged at step {step+1}, error={err*1000:.3f} mm")
            break
    else:
        # for-else: 如果循环正常结束（没有 break），执行此分支
        err = np.linalg.norm(data.site_xpos[right_sid] - target_pos)
        print(f"  IK done ({IK_STEPS} steps), residual={err*1000:.3f} mm")

    # 记录 IK 求解的终点关节角度
    qf = data.qpos.copy()
    right_final = data.site_xpos[right_sid].copy()

    print(f"  Initial: {right_init}")           # 右手起点
    print(f"  Target:  {target_pos}")           # 右手目标
    print(f"  Final:   {right_final}")          # IK 求解后右手实际位置
    print(f"  IK error: {np.linalg.norm(right_final - target_pos)*1000:.3f} mm")

    # -------------------------------------------------------------------------
    # 第6部分：S 曲线轨迹回放 + 动力学分析（ID / FD）
    # -------------------------------------------------------------------------
    # 到这里，我们知道了起点 (q0) 和终点 (qf)。
    # 接下来要做的是：
    #
    #   (a) 生成 S 曲线轨迹: 从 q0 平滑过渡到 qf，每步得到 q(t), q̇(t), q̈(t)
    #   (b) 逆动力学 (ID): 把 q/q̇/q̈ 输入 MuJoCo，算出需要的关节力矩 τ
    #   (c) 正动力学 (FD): 把 τ 施加到模型，验证加速度 q̈_FD 是否等于 q̈
    #   (d) 在 3D viewer 中播放运动，同时记录所有数据到 CSV

    n_frames = int(TRAJ_DURATION * FREQ)    # 总帧数 = 2秒 * 200Hz = 400

    # 创建独立的 MjData 用于轨迹回放（不与 IK 的 configuration 共享状态）
    data_pb = mujoco.MjData(model)

    # RateLimiter: 保证每帧真实时间 ≈ DT（让 viewer 播放速度与物理时间一致）
    rate = RateLimiter(frequency=FREQ, warn=False)

    # ---- CSV 列名 ----
    # 共 51 列: 时间 + 9 关节角度 + 9 速度 + 9 加速度 + 9 力矩
    #          + 右手 FK 位置(3) + 目标位置(3) + 位置误差
    #          + 9 个 FD 加速度误差
    csv_columns = (
        ["t"]
        + [f"q_{j}"   for j in upper_joint_names]
        + [f"qd_{j}"  for j in upper_joint_names]
        + [f"qdd_{j}" for j in upper_joint_names]
        + [f"tau_{j}" for j in upper_joint_names]
        + ["right_fk_x","right_fk_y","right_fk_z"]
        + ["right_target_x","right_target_y","right_target_z"]
        + ["right_err"]
        + [f"qdd_err_{j}" for j in upper_joint_names]
    )
    rows = []   # 存储每一帧的数据行

    # ---- 打开 3D 可视化窗口 ----
    with mujoco.viewer.launch_passive(
        model=model, data=data_pb,
        show_left_ui=False,   # 不显示左侧 UI 面板
        show_right_ui=False,  # 不显示右侧 UI 面板
    ) as viewer:
        # 设置默认自由相机视角
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # S 曲线只插值 19 个铰链关节（不包括 freejoint 的 7 个自由度）
        # 因为 freejoint（骨盆在空间中的位置+朝向）在整个运动过程中不变
        q0_hinge = q0[7:]   # 起点铰链关节角度（19 个）
        qf_hinge = qf[7:]   # 终点铰链关节角度（19 个）

        for i in range(n_frames + 1):   # +1 确保包含终点帧 (t = T)
            t = i * DT

            # (a) S 曲线采样 —— 得到当前时刻的 q, q̇, q̈
            qh, qdh, qddh = s_curve_sample(t, TRAJ_DURATION, q0_hinge, qf_hinge)

            # 组装完整的机器人状态向量
            # qpos (26 维):  7 个 freejoint（骨盆位置+四元数）+ 19 个铰链关节
            # qvel (25 维):  6 个 freejoint 速度 + 19 个铰链关节速度
            # qacc (25 维):  6 个 freejoint 加速度 + 19 个铰链关节加速度
            data_pb.qpos[:7] = q0[:7]     # freejoint 位置不变（骨盆固定在原地）
            data_pb.qpos[7:] = qh         # 铰链关节角度 = S 曲线当前值
            data_pb.qvel[:6] = 0.0        # 骨盆速度 = 0
            data_pb.qvel[6:] = qdh        # 铰链关节速度 = S 曲线当前值
            data_pb.qacc[:6] = 0.0        # 骨盆加速度 = 0
            data_pb.qacc[6:] = qddh       # 铰链关节加速度 = S 曲线当前值

            # -----------------------------------------------------------------
            # (b) 逆动力学 (ID): 运动 → 关节力矩
            # -----------------------------------------------------------------
            # mj_inverse 读取 data.qpos, data.qvel, data.qacc
            #        计算 data.qfrc_inverse（产生该运动所需的广义力）
            # 刚体动力学方程:  M(q)·q̈ + C(q,q̇)·q̇ + G(q) = τ
            #
            # 简单理解: 已知机器人要怎么动（q, q̇, q̈），
            #          计算每个关节需要多大的力矩（τ）才能实现这个运动
            mujoco.mj_inverse(model, data_pb)

            # qfrc_inverse: 广义力向量（25 维），包含所有关节所需的力/力矩
            #   对于自由关节（6 维）：反作用力/力矩（地面反力等）
            #   对于铰链关节（19 维）：驱动力矩（这就是我们关心的 τ）
            tau = data_pb.qfrc_inverse.copy()

            # -----------------------------------------------------------------
            # (c) 正动力学 (FD): 力矩 → 加速度（验证）
            # -----------------------------------------------------------------
            # 把 ID 算出的力矩 τ 作为输入，运行正向动力学
            #   方程: q̈ = M⁻¹(q) · (τ - C(q,q̇)·q̇ - G(q))
            #
            # 如果 ID 和 FD 之间数学自洽（它们确实是，因为使用同一物理引擎），
            # 那么 q̈_FD 应该等于 q̈（我们输入的加速度），
            # 即 qdd_err = q̈_FD - q̈ ≈ 0。

            # 保存原有的 qfrc_applied（外部施加力），之后恢复
            qfrc_save = data_pb.qfrc_applied.copy()

            # 将 ID 算出的力矩设为外部施加力
            data_pb.qfrc_applied[:] = tau

            # 运行正向动力学：给定力矩 → 计算加速度
            mujoco.mj_forward(model, data_pb)

            # 读取 FD 产生的加速度
            qdd_fd = data_pb.qacc.copy()

            # 恢复原有外部力（避免影响后续步骤）
            data_pb.qfrc_applied[:] = qfrc_save

            # -----------------------------------------------------------------
            # 正向运动学 (FK): 关节角度 → 末端位置
            # -----------------------------------------------------------------
            # forward 之后, site_xpos 已自动更新
            # 读取右手腕部 site 的世界坐标
            right_fk = data_pb.site_xpos[right_sid].copy()

            # 当前位置到目标位置的距离（衡量 IK 精度）
            pos_err = np.linalg.norm(right_fk - target_pos)

            # -----------------------------------------------------------------
            # 记录当前帧的所有数据
            # -----------------------------------------------------------------
            rows.append(
                [t]                                                      # 时间
                + data_pb.qpos[upper_qpos_ids].tolist()                  # 上半身 9 关节角
                + data_pb.qvel[upper_dof_ids].tolist()                   # 上半身 9 关节速度
                + data_pb.qacc[upper_dof_ids].tolist()                   # 上半身 9 关节加速度
                + tau[upper_dof_ids].tolist()                            # 上半身 9 关节力矩
                + right_fk.tolist() + target_pos.tolist()                # FK 位置 + 目标位置
                + [pos_err]                                              # 位置误差
                + (qdd_fd[upper_dof_ids] - data_pb.qacc[upper_dof_ids]).tolist()  # FD 加速度误差
            )

            # 更新灯光位置（跟随相机，让渲染好看）
            mujoco.mj_camlight(model, data_pb)

            # 同步 viewer（把当前状态推送到 3D 窗口）
            viewer.sync()

            # 等待到下一帧的时间（保证实时播放速度）
            rate.sleep()

    # -------------------------------------------------------------------------
    # 第7部分：保存数据 + 打印摘要
    # -------------------------------------------------------------------------

    # 写入 CSV 文件（逗号分隔值，可用 Excel/Python 读取）
    csv_path = os.path.join(os.path.dirname(__file__), "experiment_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)    # 写表头
        writer.writerows(rows)          # 写数据行

    # 快速统计：打印位置误差和力矩摘要
    arr = np.array(rows, dtype=np.float64)
    col = {n: i for i, n in enumerate(csv_columns)}   # 列名 → 列索引
    errs = arr[:, col["right_err"]] * 1000             # 转为毫米

    print(f"\nData saved: {csv_path}  ({len(rows)} frames)")
    print(f"Position error: mean={np.mean(errs):.3f} mm  max={np.max(errs):.3f} mm")
    print("Run plot_*.py to generate figures.")
