# -*- coding: utf-8 -*-
"""
================================================================================
 Unitree H1 上身 —— 正逆运动学 + 动力学 联合仿真演示
 基于 MuJoCo viewer 实时渲染

 ====================== 演示内容 ======================

   运动学:
   1. IK —— L-BFGS-B 数值优化求解关节角度（12起点随机搜索）
   2. FK —— MuJoCo 正向运动学实时计算末端位置，终端显示跟踪误差

   动力学:
   3. PD+重力前馈控制 —— tau = Kp*Delta_q + Kd*Delta_qd + G(q)
   4. 力矩/加速度实时显示 —— 终端显示 |tau|, |G|, |qdd|
   5. ID/FD验证（按I键） —— 运行时验证 mj_inverse 与手动公式一致性
   6. 重力补偿对比（按G键） —— 切换重力前馈 ON/OFF 观察精度差异

 ====================== 运行方式 ======================
   conda activate robotics
   python simulation_demo.py

 ====================== 键盘控制 ======================
   N      跳到下一个目标姿态
   G      切换重力补偿 ON/OFF（观察稳态误差变化）
   D      切换动力学详情显示（力矩/加速度概要）
   I      触发 ID+FD 双方法验证 + 质量矩阵属性
   SPACE  暂停/继续
   ESC    退出
================================================================================
"""

import os
import sys
import numpy as np

# ---- 屏蔽 MuJoCo 加载 STL 纹理时的 libpng sRGB 警告 ----
# 这些警告来自网格文件的色彩配置文件，不影响仿真，但会刷屏
class _LibPNGFilter:
    def __init__(self, stream):
        self.stream = stream
    def write(self, msg):
        if 'libpng warning' not in msg and 'iCCP' not in msg:
            self.stream.write(msg)
    def flush(self):
        self.stream.flush()

sys.stderr = _LibPNGFilter(sys.stderr)

import mujoco
import mujoco.viewer
import time
from scipy.optimize import minimize
from scipy import linalg

# ========================== 全局配置 ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
SCENE_FILE = "scene_upper_body.xml"

model = mujoco.MjModel.from_xml_path(SCENE_FILE)
data = mujoco.MjData(model)

# ========================== 模型索引 ==========================
JOINT_NAMES = [
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll",
    "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow",
]

joint_qposadrs = []
joint_dofadrs = []
joint_ranges = []

for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_qposadrs.append(model.jnt_qposadr[jid])
    joint_dofadrs.append(model.jnt_dofadr[jid])
    joint_ranges.append(model.jnt_range[jid].copy())

nv = model.nv

ee_left_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
ee_right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
EE_OFFSET = np.array([0.28, 0.0, -0.015])

LARM_IDX = [1, 2, 3, 4]
RARM_IDX = [5, 6, 7, 8]
TORSO_IDX = [0]


# ======================================================================
#  辅助函数
# ======================================================================

def get_q():
    return np.array([data.qpos[adr] for adr in joint_qposadrs])

def get_qd():
    return np.array([data.qvel[adr] for adr in joint_dofadrs])

def get_qdd():
    return np.array([data.qacc[adr] for adr in joint_dofadrs])

def set_torque(tau):
    data.qfrc_applied[:] = 0.0
    for i, adr in enumerate(joint_dofadrs):
        data.qfrc_applied[adr] = tau[i]

def get_ee_pos(side="left"):
    bid = ee_left_bid if side == "left" else ee_right_bid
    return data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ EE_OFFSET

def get_gravity():
    """重力力矩 G(q)，通过 mj_inverse(qacc=0, qvel=0) 求解。恢复速度不污染状态。"""
    qd_bak = get_qd().copy()
    for adr in joint_dofadrs:
        data.qvel[adr] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    G = np.array([data.qfrc_inverse[adr] for adr in joint_dofadrs])
    for i, adr in enumerate(joint_dofadrs):
        data.qvel[adr] = qd_bak[i]
    return G

def compute_mass_matrix():
    """稠密质量矩阵 M(q) —— 需在 mj_forward 之后调用以获取最新 qM"""
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M


# ======================================================================
#  动力学验证（仅在需要时调用，会暂时修改 data 状态）
# ======================================================================

def verify_id():
    """
    ID 双方法验证: mj_inverse vs M*qdd + bias
    比较当前 (q, qd, qdd) 下的两种逆动力学计算结果
    """
    q_now, qd_now, qdd_now = get_q(), get_qd(), get_qdd()
    # 方法1
    data.qacc[:] = 0.0
    for i, adr in enumerate(joint_dofadrs):
        data.qacc[adr] = qdd_now[i]
    mujoco.mj_inverse(model, data)
    tau_mj = np.array([data.qfrc_inverse[adr] for adr in joint_dofadrs])
    # 方法2
    M = compute_mass_matrix()
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    bias = np.array([data.qfrc_inverse[adr] for adr in joint_dofadrs])
    tau_man = M @ qdd_now + bias
    return np.max(np.abs(tau_mj - tau_man)), tau_mj


def verify_fd():
    """
    FD 双方法验证: mj_forward vs M^{-1}*(tau - bias)
    比较当前力矩下的两种正动力学计算结果
    """
    M = compute_mass_matrix()
    tau = np.array([data.qfrc_applied[adr] for adr in joint_dofadrs])
    qacc_mj = get_qdd()
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    bias = np.array([data.qfrc_inverse[adr] for adr in joint_dofadrs])
    qacc_man = linalg.solve(M, tau - bias, assume_a='pos')
    return np.max(np.abs(qacc_mj - qacc_man))


# ======================================================================
#  IK 求解器
# ======================================================================

def ik_solve(target_pos, side="left", use_torso=True):
    bid = ee_left_bid if side == "left" else ee_right_bid
    j_indices = (TORSO_IDX + (LARM_IDX if side == "left" else RARM_IDX)) if use_torso \
        else (LARM_IDX if side == "left" else RARM_IDX)
    bounds = [(joint_ranges[i][0], joint_ranges[i][1]) for i in j_indices]

    def cost(q_sub):
        for idx, ang in zip(j_indices, q_sub):
            data.qpos[joint_qposadrs[idx]] = ang
        mujoco.mj_forward(model, data)
        pos = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ EE_OFFSET
        # 位置误差 + 正则化（偏向零姿态，避免冗余自由度下的怪异解）
        # 权重 1e-4：足够大以阻止 q≈pi 的翻转解，足够小以允许 q≈0.3 的正常调节
        reg = 1e-4 * np.sum(q_sub**2)
        return np.sum((pos - target_pos) ** 2) + reg

    best_q, best_cost = None, np.inf
    # 多起点搜索：零起点（避免冗余自由度下找到怪异解）+ 11个随机起点
    starts = [np.zeros(len(j_indices))] + \
             [np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
              for _ in range(11)]
    for q0 in starts:
        res = minimize(cost, q0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-14})
        if res.fun < best_cost:
            best_cost = res.fun
            best_q = res.x
    full_q = get_q().copy()
    for idx, ang in zip(j_indices, best_q):
        full_q[idx] = ang
    return full_q, best_cost


# ======================================================================
#  PD + 重力前馈控制器
# ======================================================================

Kp = np.array([500, 300, 300, 150, 150, 300, 300, 150, 150])
Kd = np.array([60, 30, 30, 15, 15, 30, 30, 15, 15])


def pd_control(q_des, qd_des, use_gravity_comp):
    """PD + 可选重力前馈，输出限幅 +/-150 Nm"""
    q, qd = get_q(), get_qd()
    tau = Kp * (q_des - q) + Kd * (qd_des - qd)
    if use_gravity_comp:
        tau += get_gravity()
    return np.clip(tau, -150, 150)


# ======================================================================
#  轨迹生成
# ======================================================================

def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5

def smooth_step_deriv(t):
    t = np.clip(t, 0.0, 1.0)
    return 30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4


# ======================================================================
#  演示目标
# ======================================================================

DEMO_TARGETS = [
    ("1-Home (双臂自然下垂)", {
        "left": np.array([0.30, 0.21, 1.15]),
        "right": np.array([0.30, -0.21, 1.15]),
    }, 1.5),
    ("2-右臂上举", {
        "left": np.array([0.30, 0.21, 1.15]),
        "right": np.array([0.20, -0.21, 1.35]),
    }, 2.0),
    ("3-左臂前伸", {
        "left": np.array([0.42, 0.21, 1.08]),
        "right": np.array([0.30, -0.21, 1.15]),
    }, 2.0),
    ("4-双臂前伸", {
        "left": np.array([0.42, 0.21, 1.08]),
        "right": np.array([0.42, -0.21, 1.08]),
    }, 2.5),
    ("5-T型展开 (侧平举)", {
        "left": np.array([0.25, 0.42, 1.15]),
        "right": np.array([0.25, -0.42, 1.15]),
    }, 2.5),
    ("6-左高右低 (不对称)", {
        "left": np.array([0.22, 0.28, 1.30]),
        "right": np.array([0.35, -0.28, 1.05]),
    }, 2.0),
]

# ======================================================================
#  IK 预求解
# ======================================================================

print("=" * 70)
print("  Unitree H1 上身  运动学 + 动力学  联合仿真演示")
print("=" * 70)
print("\n[IK预求解] L-BFGS-B + 12起点随机搜索")

ik_results = []
for name, targets, _ in DEMO_TARGETS:
    # 默认不用躯干 (use_torso=False) —— 手臂4DOF对3D任务已足够
    # 双臂独立求解，避免共享躯干角度冲突
    use_t = "躯干" in name
    q_left, err_l = ik_solve(targets["left"], "left", use_torso=use_t)
    for i, adr in enumerate(joint_qposadrs):
        data.qpos[adr] = q_left[i]
    q_full, err_r = ik_solve(targets["right"], "right", use_torso=use_t)
    ik_results.append(q_full)
    print(f"  {name:35s}  IK: L={err_l*1000:.1f}mm R={err_r*1000:.1f}mm  "
          f"torso={q_full[0]:.2f} Lelb={q_full[4]:.2f} Relb={q_full[8]:.2f}{' (含躯干)' if use_t else ''}")

# 重置到 Home
for adr in joint_qposadrs:
    data.qpos[adr] = 0.0
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

# ========================== 仿真状态 ==========================

current_target_idx = 0
q_start = np.zeros(nv)
q_goal = ik_results[0]
traj_time = 0.0
traj_duration = 2.0
hold_time = 0.0
hold_duration = DEMO_TARGETS[0][2]
phase = "moving"

gravity_comp = True
show_dynamics = True
paused = False

print(f"\n[仿真就绪]")
print(f"  键盘: N=下一目标  G=重力补偿  D=动力学详情  I=ID/FD验证  SPACE=暂停  ESC=退出")
print(f"  重力补偿: {'ON' if gravity_comp else 'OFF'}  |  动力学: {'ON' if show_dynamics else 'OFF'}")
print("=" * 70)

# ========================== GLFW 键盘回调 ==========================

key_callback_data = {
    "next_target": False, "toggle_grav": False,
    "toggle_dyn": False, "run_verify": False, "paused": False
}

try:
    import glfw

    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_N:
                key_callback_data["next_target"] = True
            elif key == glfw.KEY_G:
                key_callback_data["toggle_grav"] = True
            elif key == glfw.KEY_D:
                key_callback_data["toggle_dyn"] = True
            elif key == glfw.KEY_I:
                key_callback_data["run_verify"] = True
            elif key == glfw.KEY_SPACE:
                key_callback_data["paused"] = not key_callback_data["paused"]
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    HAS_KEYBOARD = True
except (ImportError, AttributeError):
    HAS_KEYBOARD = False


# ======================================================================
#  主仿真循环
# ======================================================================

try:
    with mujoco.viewer.launch_passive(model, data,
                                       show_left_ui=False,
                                       show_right_ui=False) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 170
        viewer.cam.elevation = -15
        viewer.cam.lookat[:] = [0.15, 0.0, 1.2]

        if HAS_KEYBOARD:
            try:
                glfw.set_key_callback(viewer.window, key_callback)
            except Exception:
                HAS_KEYBOARD = False

        step = 0
        last_print = 0.0

        while viewer.is_running():
            step += 1
            sim_t = step * model.opt.timestep

            # ===== 键盘处理 =====
            if HAS_KEYBOARD:
                if key_callback_data["next_target"]:
                    key_callback_data["next_target"] = False
                    current_target_idx = (current_target_idx + 1) % len(DEMO_TARGETS)
                    name, targets, hold_d = DEMO_TARGETS[current_target_idx]
                    q_start = get_q().copy()
                    q_goal = ik_results[current_target_idx]
                    traj_time = 0.0; hold_time = 0.0
                    hold_duration = hold_d; phase = "moving"
                    print(f"\n  >> 切换: {name}")

                if key_callback_data["toggle_grav"]:
                    key_callback_data["toggle_grav"] = False
                    gravity_comp = not gravity_comp
                    print(f"\n  >> 重力补偿: {'ON' if gravity_comp else 'OFF'}")

                if key_callback_data["toggle_dyn"]:
                    key_callback_data["toggle_dyn"] = False
                    show_dynamics = not show_dynamics
                    print(f"\n  >> 动力学详情: {'ON' if show_dynamics else 'OFF'}")

                if key_callback_data["run_verify"]:
                    key_callback_data["run_verify"] = False
                    # ----- 动力学验证（会短暂修改 data 状态）-----
                    id_err, _ = verify_id()
                    fd_err = verify_fd()
                    M = compute_mass_matrix()
                    print(f"\n  ====== 动力学验证 ======")
                    print(f"  ID: mj_inverse vs M*qdd+bias  max_err={id_err:.2e} Nm  "
                          f"{'[OK]' if id_err < 1e-8 else '[FAIL]'}")
                    print(f"  FD: mj_forward vs M^{-1}(tau-bias)  max_err={fd_err:.2e} rad/s^2  "
                          f"{'[OK]' if fd_err < 1e-8 else '[FAIL]'}")
                    print(f"  M(q): cond={np.linalg.cond(M):.1f}  det={np.linalg.det(M):.4e}  "
                          f"正定={'YES' if np.all(np.linalg.eigvalsh(M) > 0) else 'NO'}")
                    print(f"  ==========================")

                paused = key_callback_data["paused"]

            # ===== 状态机 =====
            if not paused:
                if phase == "moving":
                    traj_time += model.opt.timestep
                    if traj_time >= traj_duration:
                        traj_time = traj_duration
                        phase = "holding"
                        hold_time = 0.0
                        print(f"\n  >> 已到达: {DEMO_TARGETS[current_target_idx][0]}")
                elif phase == "holding":
                    hold_time += model.opt.timestep
                    if hold_time >= hold_duration:
                        current_target_idx = (current_target_idx + 1) % len(DEMO_TARGETS)
                        name, targets, hold_d = DEMO_TARGETS[current_target_idx]
                        q_start = get_q().copy()
                        q_goal = ik_results[current_target_idx]
                        traj_time = 0.0; hold_time = 0.0
                        hold_duration = hold_d; phase = "moving"
                        print(f"\n  >> 过渡到: {name}")

            # ===== PD + 重力前馈控制 =====
            if not paused:
                s = traj_time / traj_duration if traj_duration > 0 else 1.0
                q_des = q_start + smooth_step(s) * (q_goal - q_start)
                ds = smooth_step_deriv(s) / traj_duration if traj_duration > 0 else 0.0
                qd_des = (q_goal - q_start) * ds
                tau = pd_control(q_des, qd_des, gravity_comp)
                set_torque(tau)

            # ===== MuJoCo 物理步进 =====
            mujoco.mj_step(model, data)

            # ===== 终端显示 =====
            if sim_t - last_print >= 0.5:
                last_print = sim_t
                ee_l = get_ee_pos("left")
                ee_r = get_ee_pos("right")
                name = DEMO_TARGETS[current_target_idx][0]
                tgt_l = DEMO_TARGETS[current_target_idx][1]["left"]
                tgt_r = DEMO_TARGETS[current_target_idx][1]["right"]
                err_l = np.linalg.norm(ee_l - tgt_l) * 1000
                err_r = np.linalg.norm(ee_r - tgt_r) * 1000
                phase_str = "暂停" if paused else ("跟踪" if phase == "moving" else "停留")
                grav_str = "ON" if gravity_comp else "OFF"
                progress = traj_time / traj_duration * 100 if traj_duration > 0 else 100

                # 主显示行
                q_now = get_q()
                jerr = np.linalg.norm(q_goal - q_now) * 1000  # 关节误差 mrad
                line = (f"\r  [{phase_str}] {name:35s}  "
                        f"进度:{progress:5.1f}%  "
                        f"左EE:{err_l:4.0f}mm  右EE:{err_r:4.0f}mm  "
                        f"|q_err|={jerr:5.0f}mrad  "
                        f"Grav:{grav_str}")
                # 动力学附加
                if show_dynamics and not paused:
                    t = np.array([data.qfrc_applied[adr] for adr in joint_dofadrs])
                    G = get_gravity()
                    qdd = get_qdd()
                    line += (f"  |  |tau|={np.linalg.norm(t):5.1f}  "
                             f"|G|={np.linalg.norm(G):5.1f}  "
                             f"|qdd|={np.linalg.norm(qdd):5.1f}")
                print(line, end="")

            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)

except mujoco.FatalError as e:
    print(f"\nMuJoCo错误: {e}")
    print("请确认已安装MuJoCo且支持GUI渲染（需OpenGL环境）")

print("\n\n仿真结束。")
