# -*- coding: utf-8 -*-
"""
H1 上身  双臂画圆  ——  纯 MuJoCo 仿真 (无 Pinocchio)
============================================================
  与 p_main.py 功能等价, 但不再依赖 Pinocchio 库:
    - 模型          :  scene_upper_body.xml (include h1_upper_body.xml)
    - 运动学逆解 (IK) :  MuJoCo mj_jacBody 阻尼最小二乘法
    - 动力学逆解 (ID) :  MuJoCo mj_inverse (替代 Pinocchio RNEA)
    - 轨迹生成       :  离线预计算关节空间圆周路径, 在线 PD 跟踪
    - 末端轨迹显示   :  MuJoCo viewer 中渲染红/蓝色拖尾小球

  原理:
    Pinocchio RNEA:  τ = M(q)·q̈ + C(q,v)·v + g(q)
    MuJoCo mj_inverse:  设置 qacc = q̈_des 后调用 mj_inverse,
                      qfrc_inverse = M(q)·q̈ + C(q,v)·v + g(q)
    两者在数学上等价, 均求解逆动力学。

  运行方式:
    conda activate robotics
    python two_hand_circle.py
============================================================
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from collections import deque

# ============================================================
# 1. 路径与全局配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

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
    print("预计算圆形轨迹...")

    print("  求解圆心姿态...")
    q_center = dual_ik(mj_model, mj_data, center_L, center_R,
                        q_init=np.zeros(mj_model.nv), max_iter=1000)
    set_q(mj_model, mj_data, q_center)
    eL = np.linalg.norm(get_ee_pos(mj_model, mj_data, "left") - center_L)
    eR = np.linalg.norm(get_ee_pos(mj_model, mj_data, "right") - center_R)
    print(f"  圆心 IK: eL={eL:.4f}m  eR={eR:.4f}m")

    q_traj = []
    q_prev = q_center

    for k in range(num_waypoints):
        ang = 2 * np.pi * k / num_waypoints
        tL = center_L + np.array([0., radius * np.cos(ang), radius * np.sin(ang)])
        tR = center_R + np.array([0., radius * np.cos(ang + np.pi),
                                   radius * np.sin(ang + np.pi)])

        q = dual_ik(mj_model, mj_data, tL, tR, q_init=q_prev, max_iter=300)
        q_traj.append(np.copy(q))
        q_prev = q

        if k % 12 == 0:
            eL = np.linalg.norm(get_ee_pos(mj_model, mj_data, "left") - tL)
            eR = np.linalg.norm(get_ee_pos(mj_model, mj_data, "right") - tR)
            print(f"  wp {k:3d}/{num_waypoints}  eL={eL:.4f}m  eR={eR:.4f}m")

    return np.array(q_traj)


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
    z_offset = 0.20
    center_L = pL0 + np.array([0.05, 0.04, z_offset])
    center_R = pR0 + np.array([0.05, -0.04, z_offset])
    radius = 0.06
    print(f"圆心 L: {np.round(center_L, 3)}  圆心 R: {np.round(center_R, 3)}  半径: {radius}")

    # ------- 5.4 离线预计算关节空间轨迹 -------
    q_traj = precompute_circle_trajectory(mj_model, mj_data,
                                           center_L, center_R, radius,
                                           num_waypoints=72)
    N_wp = len(q_traj)
    print(f"轨迹预计算完成: {N_wp} 个路径点")

    # ------- 5.5 PD 控制器增益 -------
    Kp = np.array([500, 300, 300, 150, 150, 300, 300, 150, 150])
    Kd = np.array([60,  30,  30,  15,  15,  30,  30,  15,  15])
    dt = mj_model.opt.timestep
    freq = 0.5

    # ------- 5.6 初始化到轨迹起点 -------
    set_q(mj_model, mj_data, q_traj[0])
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    t = 0.0

    # ------- 5.7 启动 MuJoCo viewer -------
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 170
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0.15, 0.0, 0.5]

        trail_L = deque(maxlen=200)
        trail_R = deque(maxlen=200)
        trail_tick = 0

        # ------- 5.8 仿真主循环 -------
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

            # (c) PD 控制 + MuJoCo 逆动力学补偿
            # a_pd = Kp·(q_des - q) + Kd·(0 - v)
            a_pd = Kp * (q_des - q_curr) + Kd * (0.0 - v_curr)

            # 将期望加速度写入 qacc
            for i, adr in enumerate(joint_dof_adrs):
                mj_data.qacc[adr] = a_pd[i]

            # 清零外部力, 保证 mj_inverse 计算完整的 M·qacc + bias
            mj_data.qfrc_applied[:] = 0.0

            # 逆动力学: qfrc_inverse = M(q)·qacc + C(q,v)·v + g(q)
            mujoco.mj_inverse(mj_model, mj_data)

            # 将逆动力学结果作为控制力矩施加
            mj_data.qfrc_applied[:] = 0.0
            for adr in joint_dof_adrs:
                mj_data.qfrc_applied[adr] = mj_data.qfrc_inverse[adr]

            # (d) MuJoCo 物理步进
            mujoco.mj_step(mj_model, mj_data)

            # (e) 采样末端轨迹
            trail_tick += 1
            if trail_tick % 3 == 0:
                trail_L.append(get_ee_pos(mj_model, mj_data, "left").copy())
                trail_R.append(get_ee_pos(mj_model, mj_data, "right").copy())

            # (f) 渲染拖尾
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

            # (g) 同步 viewer 并推进时间
            viewer.sync()
            t += dt

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == '__main__':
    main()
