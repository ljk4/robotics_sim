# -*- coding: utf-8 -*-
"""
H1 上身  双臂画圆
============================================================
    功能描述:
    - 模型          :  scene_upper_body.xml (include h1_upper_body.xml)
    - 运动学逆解 (IK) :  MuJoCo mj_jacBody 阻尼最小二乘法
    - 动力学逆解 (ID) :  MuJoCo mj_inverse
    - 轨迹生成       :  离线预计算关节空间圆周路径, 在线 PD 跟踪
    - 末端轨迹显示   :  MuJoCo viewer 中渲染红/蓝色拖尾小球

  运行方式:
    conda activate robotics
    python two_hand_circle.py
============================================================
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
from collections import deque

# ============================================================
# 1. 路径与全局配置
# ============================================================

XML_PATH = 'scene_upper_body.xml'

EE_OFFSET = np.array([0.28, 0.0, -0.015])

JOINT_NAMES = [
    "torso",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]
LARM_IDX = [1, 2, 3, 4]
RARM_IDX = [5, 6, 7, 8]


# ============================================================
# 2. 工具函数
# ============================================================

def get_q(mj_model, mj_data):
    """从 MuJoCo qpos 读取 9 维关节角 (rad)"""
    q = np.zeros(mj_model.nv)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[i] = mj_data.qpos[mj_model.jnt_qposadr[jid]]
    return q


def set_q(mj_model, mj_data, q):
    """写入 9 维关节角并更新正向运动学"""
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mj_data.qpos[mj_model.jnt_qposadr[jid]] = q[i]
    mujoco.mj_forward(mj_model, mj_data)


def get_ee_pos(mj_model, mj_data, side="left"):
    """末端执行器在世界坐标系中的位置 (肘连杆原点 + 前臂偏移)"""
    body_name = f"{side}_elbow_link"
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return mj_data.xpos[bid] + mj_data.xmat[bid].reshape(3, 3) @ EE_OFFSET


# ============================================================
# 3. MuJoCo IK 求解器 —— 雅可比阻尼最小二乘法
# ============================================================

def ik_mujoco(mj_model, mj_data, target_pos, side="left",
              q_init=None, max_iter=500, tol=1e-6, damping=0.1):
    """
    单臂逆向运动学求解 (阻尼最小二乘法)

        Δq = J^T (J J^T + λ²I)^{-1} · e

    参数:
      target_pos : 目标末端位置 (3,) 世界坐标
      side       : "left" / "right"
      q_init     : 初始关节角 (9,) 或 None
      damping    : 阻尼因子 λ
    """
    body_name = f"{side}_elbow_link"
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_indices = LARM_IDX if side == "left" else RARM_IDX
    n_joints = len(joint_indices)

    q = np.zeros(mj_model.nv) if q_init is None else np.copy(q_init)
    set_q(mj_model, mj_data, q)
    q_sub = np.array([q[i] for i in joint_indices])

    for _ in range(max_iter):
        set_q(mj_model, mj_data, q)

        ee_pos = get_ee_pos(mj_model, mj_data, side)
        error = target_pos - ee_pos
        if np.linalg.norm(error) < tol:
            break

        jacp = np.zeros((3, mj_model.nv))
        jacr = np.zeros((3, mj_model.nv))
        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, bid)

        r_world = mj_data.xmat[bid].reshape(3, 3) @ EE_OFFSET
        rx, ry, rz = r_world
        r_skew = np.array([[0, -rz, ry],
                           [rz, 0, -rx],
                           [-ry, rx, 0]])

        J = np.zeros((3, n_joints))
        for col, idx in enumerate(joint_indices):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT,
                                     JOINT_NAMES[idx])
            dof_adr = mj_model.jnt_dofadr[jid]
            J[:, col] = jacp[:, dof_adr] - r_skew @ jacr[:, dof_adr]

        JJT = J @ J.T
        dq_sub = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(3), error)
        q_sub += dq_sub

        for i, idx in enumerate(joint_indices):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT,
                                     JOINT_NAMES[idx])
            lo, hi = mj_model.jnt_range[jid]
            q_sub[i] = np.clip(q_sub[i], lo, hi)
            q[idx] = q_sub[i]

    return q


def dual_ik(mj_model, mj_data, target_L, target_R, q_init=None, max_iter=500):
    """双臂 IK: 先左后右, 各自独立求解 (不使用躯干)"""
    q = np.copy(q_init) if q_init is not None else np.zeros(mj_model.nv)
    q = ik_mujoco(mj_model, mj_data, target_L, "left",  q_init=q, max_iter=max_iter)
    q = ik_mujoco(mj_model, mj_data, target_R, "right", q_init=q, max_iter=max_iter)
    return q


# ============================================================
# 4. 预计算圆形轨迹
# ============================================================

def precompute_circle_trajectory(mj_model, mj_data, center_L, center_R,
                                  radius, num_waypoints=72):
    """
    离线预计算双臂圆周运动的关节空间轨迹

    圆在 YZ 平面: X 固定, Y = r·cos(θ), Z = r·sin(θ)
    左臂相位 0, 右臂相位 π (交替画圆)
    """
    print("=" * 60)
    print("  预计算圆形轨迹")
    print("=" * 60)

    print("\n  [步骤1] 求解圆心姿态 (max_iter=1000)...")
    q_center = dual_ik(mj_model, mj_data, center_L, center_R,
                        q_init=np.zeros(mj_model.nv), max_iter=1000)
    set_q(mj_model, mj_data, q_center)
    eL_c = np.linalg.norm(get_ee_pos(mj_model, mj_data, "left") - center_L)
    eR_c = np.linalg.norm(get_ee_pos(mj_model, mj_data, "right") - center_R)
    print(f"  圆心 IK 误差:  左={eL_c*1000:.2f}mm  右={eR_c*1000:.2f}mm")

    print(f"\n  [步骤2] 逐点求解圆周路径点 (max_iter=300, warm-start)...")
    print(f"  {'k':>4s}  {'ang':>7s}  "
          f"{'IK_err_L':>10s}  {'IK_err_R':>10s}  "
          f"{'|q-q_prev|':>12s}  {'注意':<s}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

    q_traj = []
    q_prev = q_center
    ik_errors_L = []
    ik_errors_R = []

    for k in range(num_waypoints):
        ang = 2 * np.pi * k / num_waypoints
        tL = center_L + np.array([0., radius * np.cos(ang), radius * np.sin(ang)])
        tR = center_R + np.array([0., radius * np.cos(ang + np.pi),
                                   radius * np.sin(ang + np.pi)])

        q = dual_ik(mj_model, mj_data, tL, tR, q_init=q_prev, max_iter=300)
        q_traj.append(np.copy(q))

        eL = np.linalg.norm(get_ee_pos(mj_model, mj_data, "left") - tL)
        eR = np.linalg.norm(get_ee_pos(mj_model, mj_data, "right") - tR)
        ik_errors_L.append(eL)
        ik_errors_R.append(eR)

        dq_norm = np.linalg.norm(q - q_prev)
        q_prev = q

        flag = " *** 大误差!" if max(eL, eR) > 0.01 else ""
        print(f"  {k:4d}  {np.degrees(ang):6.1f}°  "
              f"{eL*1000:8.2f}mm  {eR*1000:8.2f}mm  "
              f"{dq_norm:10.4f}rad{flag}")

    # ---- 汇总统计 ----
    eL_arr = np.array(ik_errors_L)
    eR_arr = np.array(ik_errors_R)
    print(f"\n  [IK误差汇总]")
    print(f"  左臂: max={eL_arr.max()*1000:.2f}mm  mean={eL_arr.mean()*1000:.2f}mm  "
          f"std={eL_arr.std()*1000:.2f}mm  >10mm的帧: {np.sum(eL_arr > 0.01)}")
    print(f"  右臂: max={eR_arr.max()*1000:.2f}mm  mean={eR_arr.mean()*1000:.2f}mm  "
          f"std={eR_arr.std()*1000:.2f}mm  >10mm的帧: {np.sum(eR_arr > 0.01)}")

    if np.sum(eL_arr > 0.01) > 0:
        bad = np.where(eL_arr > 0.01)[0]
        print(f"  左臂大误差帧: {list(bad)}")
    if np.sum(eR_arr > 0.01) > 0:
        bad = np.where(eR_arr > 0.01)[0]
        print(f"  右臂大误差帧: {list(bad)}")

    q_traj = np.array(q_traj)

    # ---- 步骤3: 检查并修复闭合点不连续 ----
    # q_traj[0] 来自 q_center warm-start, q_traj[-1] 来自 q_traj[-2] 链,
    # 两者可能落在不同的 IK 分支, 导致关节空间轨迹不闭合
    dq_wrap = np.linalg.norm(q_traj[0] - q_traj[-1])
    joint_labels = ["torso", "L_sp", "L_sr", "L_sy", "L_eb",
                    "R_sp", "R_sr", "R_sy", "R_eb"]
    print(f"\n  [步骤3] 闭合点检查: |q[0] - q[-1]| = {dq_wrap:.4f} rad")
    if dq_wrap > 0.05:
        print(f"  *** 关节空间不连续! 逐关节差异:")
        for i, label in enumerate(joint_labels):
            diff = abs(q_traj[0, i] - q_traj[-1, i])
            flag = " ***" if diff > 0.05 else ""
            print(f"      {label}: q[0]={q_traj[0,i]:.4f}  q[-1]={q_traj[-1,i]:.4f}  "
                  f"diff={diff:.4f}{flag}")

        # 用 q_traj[-1] 作为 warm-start 重新求解 q_traj[0]
        print(f"  修复: 用 q[-1] warm-start 重解 q[0]...")
        ang = 0.0
        tL = center_L + np.array([0., radius * np.cos(ang), radius * np.sin(ang)])
        tR = center_R + np.array([0., radius * np.cos(ang + np.pi),
                                   radius * np.sin(ang + np.pi)])
        q_new0 = dual_ik(mj_model, mj_data, tL, tR, q_init=q_traj[-1], max_iter=1000)
        dq_new = np.linalg.norm(q_new0 - q_traj[-1])
        print(f"    |new_q[0] - q[-1]| = {dq_new:.4f} rad  "
              f"|new - old| = {np.linalg.norm(q_new0 - q_traj[0]):.4f} rad  "
              f"闭合修复成功!" if dq_new < 0.1 else "仍有间隙")
        q_traj[0] = q_new0

        # 检查 q[0]→q[1] 是否平滑
        dq_01 = np.linalg.norm(q_traj[0] - q_traj[1])
        if dq_01 > 0.1:
            ang = 2 * np.pi * 1 / num_waypoints
            tL = center_L + np.array([0., radius * np.cos(ang), radius * np.sin(ang)])
            tR = center_R + np.array([0., radius * np.cos(ang + np.pi),
                                       radius * np.sin(ang + np.pi)])
            q_traj[1] = dual_ik(mj_model, mj_data, tL, tR, q_init=q_traj[0], max_iter=300)
            print(f"    wp[1] 已重解, |q[0]-q[1]|={np.linalg.norm(q_traj[0]-q_traj[1]):.4f}")
    else:
        print(f"  闭合点连续, 无需修复")

    return q_traj


def analyze_cartesian_path(mj_model, mj_data, q_traj, center_L, center_R, radius):
    """
    分析关节空间轨迹对应的实际笛卡尔路径质量
    - 检查每个 waypoint 的实际末端位置偏离理想圆的程度
    - 检查关节空间线性插值带来的中间点偏离
    """
    print("\n" + "=" * 60)
    print("  笛卡尔路径质量分析")
    print("=" * 60)

    N_wp = len(q_traj)

    # ---- 分析1: waypoint 的实际笛卡尔位置 ----
    actual_L = np.zeros((N_wp, 3))
    actual_R = np.zeros((N_wp, 3))
    ideal_L = np.zeros((N_wp, 3))
    ideal_R = np.zeros((N_wp, 3))

    for k in range(N_wp):
        set_q(mj_model, mj_data, q_traj[k])
        actual_L[k] = get_ee_pos(mj_model, mj_data, "left")
        actual_R[k] = get_ee_pos(mj_model, mj_data, "right")

        ang = 2 * np.pi * k / N_wp
        ideal_L[k] = center_L + np.array([0., radius * np.cos(ang), radius * np.sin(ang)])
        ideal_R[k] = center_R + np.array([0., radius * np.cos(ang + np.pi),
                                           radius * np.sin(ang + np.pi)])

    # 每个 waypoint 到理想圆心的距离
    dist_L = np.linalg.norm(actual_L - center_L, axis=1)
    dist_R = np.linalg.norm(actual_R - center_R, axis=1)
    radial_err_L = np.abs(dist_L - radius)
    radial_err_R = np.abs(dist_R - radius)

    print(f"\n  [waypoint 径向误差] (实际末端到圆心距离 - 理想半径)")
    print(f"  左臂: max={radial_err_L.max()*1000:.2f}mm  mean={radial_err_L.mean()*1000:.2f}mm  "
          f"std={radial_err_L.std()*1000:.2f}mm")
    print(f"  右臂: max={radial_err_R.max()*1000:.2f}mm  mean={radial_err_R.mean()*1000:.2f}mm  "
          f"std={radial_err_R.std()*1000:.2f}mm")

    # 3D 位置误差 (到理想圆上最近点的欧氏距离)
    pos_err_L = np.linalg.norm(actual_L - ideal_L, axis=1)
    pos_err_R = np.linalg.norm(actual_R - ideal_R, axis=1)
    print(f"\n  [waypoint 3D位置误差]")
    print(f"  左臂: max={pos_err_L.max()*1000:.2f}mm  mean={pos_err_L.mean()*1000:.2f}mm  "
          f"std={pos_err_L.std()*1000:.2f}mm")
    print(f"  右臂: max={pos_err_R.max()*1000:.2f}mm  mean={pos_err_R.mean()*1000:.2f}mm  "
          f"std={pos_err_R.std()*1000:.2f}mm")

    # ---- 分析2: 关节空间线性插值带来的中间点偏离 ----
    # 在每个相邻 waypoint 之间插值 5 个中间点, 计算偏离
    interp_err_L = []
    interp_err_R = []
    for k in range(N_wp):
        q0 = q_traj[k]
        q1 = q_traj[(k + 1) % N_wp]
        ang0 = 2 * np.pi * k / N_wp
        ang1 = 2 * np.pi * (k + 1) / N_wp
        for frac in [0.2, 0.4, 0.6, 0.8]:
            q_mid = (1 - frac) * q0 + frac * q1
            set_q(mj_model, mj_data, q_mid)
            eeL = get_ee_pos(mj_model, mj_data, "left")
            eeR = get_ee_pos(mj_model, mj_data, "right")
            ang_mid = (1 - frac) * ang0 + frac * ang1
            idealL = center_L + np.array([0., radius * np.cos(ang_mid), radius * np.sin(ang_mid)])
            idealR = center_R + np.array([0., radius * np.cos(ang_mid + np.pi),
                                           radius * np.sin(ang_mid + np.pi)])
            interp_err_L.append(np.linalg.norm(eeL - idealL))
            interp_err_R.append(np.linalg.norm(eeR - idealR))

    interp_err_L = np.array(interp_err_L)
    interp_err_R = np.array(interp_err_R)
    print(f"\n  [关节空间线性插值中间点误差] (每段插4个中间点)")
    print(f"  左臂: max={interp_err_L.max()*1000:.2f}mm  mean={interp_err_L.mean()*1000:.2f}mm  "
          f"std={interp_err_L.std()*1000:.2f}mm")
    print(f"  右臂: max={interp_err_R.max()*1000:.2f}mm  mean={interp_err_R.mean()*1000:.2f}mm  "
          f"std={interp_err_R.std()*1000:.2f}mm")

    # 找出插值后最大的偏差对应的 waypoint 区间
    worst_L_idx = np.argmax(interp_err_L)
    worst_L_seg = worst_L_idx // 4  # 每个 segment 有 4 个插值点
    print(f"  左臂最大插值偏差出现在 waypoint[{worst_L_seg}]→[{worst_L_seg+1}] 之间")
    worst_R_idx = np.argmax(interp_err_R)
    worst_R_seg = worst_R_idx // 4
    print(f"  右臂最大插值偏差出现在 waypoint[{worst_R_seg}]→[{worst_R_seg+1}] 之间")

    # 打印前10个最差的 waypoint
    print(f"\n  [左臂 - 径向误差最大的10个 waypoint]")
    worst10_L = np.argsort(radial_err_L)[::-1][:10]
    for rank, idx in enumerate(worst10_L):
        ang = np.degrees(2 * np.pi * idx / N_wp)
        print(f"    #{rank+1}: wp[{idx:3d}] @ {ang:6.1f}°  "
              f"径向误差={radial_err_L[idx]*1000:.2f}mm  "
              f"实际距离={dist_L[idx]:.4f}m (理想={radius:.4f}m)")

    print(f"\n  [右臂 - 径向误差最大的10个 waypoint]")
    worst10_R = np.argsort(radial_err_R)[::-1][:10]
    for rank, idx in enumerate(worst10_R):
        ang = np.degrees(2 * np.pi * idx / N_wp)
        print(f"    #{rank+1}: wp[{idx:3d}] @ {ang:6.1f}°  "
              f"径向误差={radial_err_R[idx]*1000:.2f}mm  "
              f"实际距离={dist_R[idx]:.4f}m (理想={radius:.4f}m)")

    print()
    return actual_L, actual_R


# ============================================================
# 5. 主函数
# ============================================================

def main():
    # ------- 5.1 加载 MuJoCo 模型 -------
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data = mujoco.MjData(mj_model)

    nv = mj_model.nv
    print(f"MuJoCo: nq={mj_model.nq}, nv={nv}, nu={mj_model.nu}")

    # 预缓存关节 DOF 地址, 避免主循环中重复查找
    joint_dof_adrs = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_dof_adrs.append(mj_model.jnt_dofadr[jid])

    # ------- 5.2 获取 q=0 时的末端位置 -------
    set_q(mj_model, mj_data, np.zeros(nv))
    pL0 = get_ee_pos(mj_model, mj_data, "left")
    pR0 = get_ee_pos(mj_model, mj_data, "right")
    print(f"左末端 q=0 位置: {np.round(pL0, 3)}")
    print(f"右末端 q=0 位置: {np.round(pR0, 3)}")

    # ------- 5.3 圆形轨迹参数 -------
    z_offset = 0.15
    center_L = pL0 + np.array([0.05, 0.04, z_offset])
    center_R = pR0 + np.array([0.05, -0.04, z_offset])
    radius = 0.12
    print(f"圆心 L: {np.round(center_L, 3)}  圆心 R: {np.round(center_R, 3)}  半径: {radius}")

    # ------- 5.4 离线预计算关节空间轨迹 -------
    q_traj = precompute_circle_trajectory(mj_model, mj_data,
                                           center_L, center_R, radius,
                                           num_waypoints=172)
    N_wp = len(q_traj)
    print(f"轨迹预计算完成: {N_wp} 个路径点")

    # ------- 5.5 分析笛卡尔路径质量 -------
    analyze_cartesian_path(mj_model, mj_data, q_traj, center_L, center_R, radius)

    # ------- 5.6 PD 控制器增益 -------
    Kp = np.array([800, 500, 500, 250, 250, 500, 500, 250, 250])
    Kd = np.array([80,  50,  50,  25,  25,  50,  50,  25,  25])
    Kp_cart = 100000  # 笛卡尔位置刚度 [N/m]
    Kd_cart = 400     # 笛卡尔阻尼 [N·s/m]
    dt = mj_model.opt.timestep
    freq = 0.5

    # ------- 5.7 初始化到轨迹起点 (含初始速度) -------
    set_q(mj_model, mj_data, q_traj[0])
    v_init = freq * N_wp * (q_traj[1] - q_traj[0])
    for i, adr in enumerate(joint_dof_adrs):
        mj_data.qvel[adr] = v_init[i]
    mujoco.mj_forward(mj_model, mj_data)

    # 预缓存笛卡尔修正用到的索引
    bid_L = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    bid_R = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    larm_dofs = [joint_dof_adrs[i] for i in LARM_IDX]
    rarm_dofs = [joint_dof_adrs[i] for i in RARM_IDX]

    t = 0.0
    print_interval = 1.0   # 每 1 秒打印一次运行时状态
    last_print_time = -print_interval
    runtime_pos_err_L = []
    runtime_pos_err_R = []
    runtime_joint_err = []

    # ------- 5.8 启动 MuJoCo viewer -------
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 170
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0.15, 0.0, 0.5]

        trail_L = deque(maxlen=200)
        trail_R = deque(maxlen=200)
        trail_tick = 0

        # ------- 5.9 仿真主循环 -------
        joint_labels = ["torso", "L_sp", "L_sr", "L_sy", "L_eb",
                        "R_sp", "R_sr", "R_sy", "R_eb"]

        print("\n" + "=" * 60)
        print("  仿真运行中... (每1秒输出运行时状态)")
        print("=" * 60)
        header1 = (f"  {'t':>6s}  {'phase':>6s}  {'err_L':>8s}  {'err_R':>8s}  "
                   f"{'|jerr|':>8s}  {'|v_des|':>7s}  {'|tau|':>7s}")
        header2 = (f"  {'':6s}  {'':6s}  {'mm':>8s}  {'mm':>8s}  "
                   f"{'rad':>8s}  {'rad/s':>7s}  {'Nm':>7s}")
        print(header1)
        print(header2)
        print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

        while viewer.is_running():
            step_start = time.time()

            # (a) 读取当前状态
            q_curr = get_q(mj_model, mj_data)
            v_curr = np.array([mj_data.qvel[adr] for adr in joint_dof_adrs])

            # (b) 从预计算轨迹插值目标关节角
            phase = (freq * t) % 1.0
            idx = phase * N_wp
            i0 = int(np.floor(idx)) % N_wp
            i1 = (i0 + 1) % N_wp
            frac = idx - np.floor(idx)
            q_des = (1 - frac) * q_traj[i0] + frac * q_traj[i1]

            # (c) 计算理想末端位置
            ang = 2 * np.pi * phase
            idealL = center_L + np.array([0., radius * np.cos(ang), radius * np.sin(ang)])
            idealR = center_R + np.array([0., radius * np.cos(ang + np.pi),
                                           radius * np.sin(ang + np.pi)])

            # (d) 计算期望速度
            v_des = freq * N_wp * (q_traj[i1] - q_traj[i0])

            # (e) 关节空间 PD
            a_pd = Kp * (q_des - q_curr) + Kd * (v_des - v_curr)

            for i, adr in enumerate(joint_dof_adrs):
                mj_data.qacc[adr] = a_pd[i]

            # (f) 逆动力学
            mj_data.qfrc_applied[:] = 0.0
            mujoco.mj_inverse(mj_model, mj_data)

            # (g) 笛卡尔修正: 直接加在力矩空间
            eeL_curr = get_ee_pos(mj_model, mj_data, "left")
            eeR_curr = get_ee_pos(mj_model, mj_data, "right")
            errL = idealL - eeL_curr
            errR = idealR - eeR_curr

            mj_data.qfrc_applied[:] = 0.0
            for adr in joint_dof_adrs:
                mj_data.qfrc_applied[adr] = mj_data.qfrc_inverse[adr]

            # 左臂
            jacp = np.zeros((3, mj_model.nv)); jacr = np.zeros((3, mj_model.nv))
            mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, bid_L)
            rw = mj_data.xmat[bid_L].reshape(3, 3) @ EE_OFFSET
            rsk = np.array([[0, -rw[2], rw[1]], [rw[2], 0, -rw[0]], [-rw[1], rw[0], 0]])
            JL = np.zeros((3, 4))
            for col, da in enumerate(larm_dofs):
                JL[:, col] = jacp[:, da] - rsk @ jacr[:, da]
            v_arm_L = np.array([v_curr[i] for i in LARM_IDX])
            tau_cart_L = JL.T @ (Kp_cart * errL - Kd_cart * (JL @ v_arm_L))
            for col in range(4):
                mj_data.qfrc_applied[larm_dofs[col]] += tau_cart_L[col]

            # 右臂
            jacp = np.zeros((3, mj_model.nv)); jacr = np.zeros((3, mj_model.nv))
            mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, bid_R)
            rw = mj_data.xmat[bid_R].reshape(3, 3) @ EE_OFFSET
            rsk = np.array([[0, -rw[2], rw[1]], [rw[2], 0, -rw[0]], [-rw[1], rw[0], 0]])
            JR = np.zeros((3, 4))
            for col, da in enumerate(rarm_dofs):
                JR[:, col] = jacp[:, da] - rsk @ jacr[:, da]
            v_arm_R = np.array([v_curr[i] for i in RARM_IDX])
            tau_cart_R = JR.T @ (Kp_cart * errR - Kd_cart * (JR @ v_arm_R))
            for col in range(4):
                mj_data.qfrc_applied[rarm_dofs[col]] += tau_cart_R[col]

            # (h) MuJoCo 物理步进
            mujoco.mj_step(mj_model, mj_data)

            # (i) 运行时状态打印
            if t - last_print_time >= print_interval:
                last_print_time = t
                errL = np.linalg.norm(eeL_curr - idealL)
                errR = np.linalg.norm(eeR_curr - idealR)
                jerr_vec = q_des - q_curr
                jerr = np.linalg.norm(jerr_vec)
                tau_vec = np.array([mj_data.qfrc_applied[adr] for adr in joint_dof_adrs])
                tau_norm = np.linalg.norm(tau_vec)

                runtime_pos_err_L.append(errL)
                runtime_pos_err_R.append(errR)
                runtime_joint_err.append(jerr)

                # 汇总行
                print(f"  {t:5.1f}s  {phase:5.1%}  {errL*1000:7.2f}mm  {errR*1000:7.2f}mm  "
                      f"{jerr:7.4f}rad  {np.linalg.norm(v_des):6.2f}  {tau_norm:7.1f}Nm")
                # 逐关节误差
                jerr_str = "  ".join(f"{e*1000:5.1f}" for e in jerr_vec)  # mrad
                label_str = "  ".join(f"{l:>5s}" for l in joint_labels)
                print(f"        逐关节误差(mrad):  {label_str}")
                print(f"                           {jerr_str}")
                # 逐关节力矩
                tau_str = "  ".join(f"{t:5.1f}" for t in tau_vec)
                print(f"        逐关节力矩(Nm)  :  {tau_str}")

            # (g) 采样末端轨迹
            trail_tick += 1
            if trail_tick % 3 == 0:
                trail_L.append(get_ee_pos(mj_model, mj_data, "left").copy())
                trail_R.append(get_ee_pos(mj_model, mj_data, "right").copy())

            # (h) 渲染拖尾
            scn = viewer.user_scn
            scn.ngeom = 0

            for trail, rgba in [(trail_L, [1.0, 0.2, 0.2, 0.7]),
                                (trail_R, [0.2, 0.5, 1.0, 0.7])]:
                for pt in trail:
                    if scn.ngeom >= scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.006, 0.0, 0.0]),
                        pos=np.array(pt),
                        mat=np.eye(3).flatten(),
                        rgba=np.array(rgba),
                    )
                    scn.ngeom += 1

            # (i) 同步 viewer 并推进时间
            viewer.sync()
            t += dt

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        # ---- 仿真结束后输出运行时汇总 ----
        if len(runtime_pos_err_L) > 0:
            errL_arr = np.array(runtime_pos_err_L)
            errR_arr = np.array(runtime_pos_err_R)
            jerr_arr = np.array(runtime_joint_err)
            print(f"\n  [运行时跟踪误差汇总] ({len(errL_arr)} 个采样点)")
            print(f"  末端位置误差 - 左: max={errL_arr.max()*1000:.2f}mm  "
                  f"mean={errL_arr.mean()*1000:.2f}mm  std={errL_arr.std()*1000:.2f}mm")
            print(f"  末端位置误差 - 右: max={errR_arr.max()*1000:.2f}mm  "
                  f"mean={errR_arr.mean()*1000:.2f}mm  std={errR_arr.std()*1000:.2f}mm")
            print(f"  关节空间误差    : max={jerr_arr.max():.4f}rad  "
                  f"mean={jerr_arr.mean():.4f}rad  std={jerr_arr.std():.4f}rad")


if __name__ == '__main__':
    main()
