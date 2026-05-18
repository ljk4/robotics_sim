# -*- coding: utf-8 -*-
"""记录实际末端轨迹并分析圆度, 测试笛卡尔修正"""
import numpy as np
import mujoco
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

XML_PATH = 'scene_upper_body.xml'
EE_OFFSET = np.array([0.28, 0.0, -0.015])

JOINT_NAMES = [
    "torso", "left_shoulder_pitch", "left_shoulder_roll",
    "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow",
]
LARM_IDX = [1, 2, 3, 4]
RARM_IDX = [5, 6, 7, 8]
JLAB = ["torso", "L_sp", "L_sr", "L_sy", "L_eb",
        "R_sp", "R_sr", "R_sy", "R_eb"]


def get_q(m, d):
    q = np.zeros(m.nv)
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[i] = d.qpos[m.jnt_qposadr[jid]]
    return q


def set_q(m, d, q):
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[jid]] = q[i]
    mujoco.mj_forward(m, d)


def get_ee_pos(m, d, side="left"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")
    return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ EE_OFFSET


def ik_one(m, d, target, side, q_init, max_iter=500, tol=1e-6, damping=0.1):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")
    jidx = LARM_IDX if side == "left" else RARM_IDX
    q = np.copy(q_init)
    set_q(m, d, q)
    q_sub = np.array([q[i] for i in jidx])
    for _ in range(max_iter):
        set_q(m, d, q)
        e = target - get_ee_pos(m, d, side)
        if np.linalg.norm(e) < tol:
            break
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, bid)
        rw = d.xmat[bid].reshape(3, 3) @ EE_OFFSET
        rx, ry, rz = rw
        rsk = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
        J = np.zeros((3, len(jidx)))
        for c, idx in enumerate(jidx):
            da = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[idx])]
            J[:, c] = jp[:, da] - rsk @ jr[:, da]
        dq_sub = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), e)
        q_sub += dq_sub
        for c, idx in enumerate(jidx):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[idx])
            lo, hi = m.jnt_range[jid]
            q_sub[c] = np.clip(q_sub[c], lo, hi)
            q[idx] = q_sub[c]
    return q


def dual_ik(m, d, tL, tR, qi, max_iter=500):
    q = np.copy(qi)
    q = ik_one(m, d, tL, "left",  q, max_iter)
    q = ik_one(m, d, tR, "right", q, max_iter)
    return q


def precompute(m, d, cL, cR, r, nw=172):
    qc = dual_ik(m, d, cL, cR, np.zeros(m.nv), max_iter=1000)
    qt, qp = [], qc
    for k in range(nw):
        ang = 2 * np.pi * k / nw
        tL = cL + np.array([0., r * np.cos(ang), r * np.sin(ang)])
        tR = cR + np.array([0., r * np.cos(ang + np.pi), r * np.sin(ang + np.pi)])
        q = dual_ik(m, d, tL, tR, qp, max_iter=300)
        qt.append(np.copy(q))
        qp = q
    qt = np.array(qt)
    # 闭合点修复
    dw = np.linalg.norm(qt[0] - qt[-1])
    if dw > 0.05:
        qt[0] = dual_ik(m, d, cL + np.array([0., r, 0.]),
                         cR + np.array([0., -r, 0.]), qt[-1], max_iter=1000)
        if np.linalg.norm(qt[0] - qt[1]) > 0.1:
            a2 = 2 * np.pi / nw
            qt[1] = dual_ik(m, d,
                             cL + np.array([0., r*np.cos(a2), r*np.sin(a2)]),
                             cR + np.array([0., r*np.cos(a2+np.pi), r*np.sin(a2+np.pi)]),
                             qt[0], max_iter=300)
    return qt


def simulate(m, d, qt, cL, cR, r, use_cart=True, freq=0.5, duration=2.5):
    """运行仿真, use_cart=True 时启用笛卡尔修正"""
    N_wp = len(qt)
    dt = m.opt.timestep
    dof = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES]

    Kp = np.array([800, 500, 500, 250, 250, 500, 500, 250, 250])
    Kd = np.array([80,  50,  50,  25,  25,  50,  50,  25,  25])
    Kp_cart = 100000
    Kd_cart = 400

    set_q(m, d, qt[0])
    vi = freq * N_wp * (qt[1] - qt[0])
    for i, ad in enumerate(dof):
        d.qvel[ad] = vi[i]
    mujoco.mj_forward(m, d)

    bid_L = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    bid_R = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    larm_dofs = [dof[i] for i in LARM_IDX]
    rarm_dofs = [dof[i] for i in RARM_IDX]

    t = 0.0
    steps = int(duration / dt)
    n_s = steps + 1
    eeL = np.zeros((n_s, 3))
    eeR = np.zeros((n_s, 3))
    times = np.zeros(n_s)

    for step in range(n_s):
        times[step] = t
        eeL[step] = get_ee_pos(m, d, "left")
        eeR[step] = get_ee_pos(m, d, "right")

        if step == steps:
            break

        qc = get_q(m, d)
        vc = np.array([d.qvel[ad] for ad in dof])
        ph = (freq * t) % 1.0
        idx = ph * N_wp
        i0 = int(np.floor(idx)) % N_wp
        i1 = (i0 + 1) % N_wp
        fc = idx - np.floor(idx)
        qd = (1 - fc) * qt[i0] + fc * qt[i1]
        vd = freq * N_wp * (qt[i1] - qt[i0])
        apd = Kp * (qd - qc) + Kd * (vd - vc)

        if use_cart:
            ang = 2 * np.pi * ph
            idL = cL + np.array([0., r * np.cos(ang), r * np.sin(ang)])
            idR = cR + np.array([0., r * np.cos(ang + np.pi), r * np.sin(ang + np.pi)])

            eL = idL - eeL[step]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, bid_L)
            rw = d.xmat[bid_L].reshape(3, 3) @ EE_OFFSET
            rs = np.array([[0, -rw[2], rw[1]], [rw[2], 0, -rw[0]], [-rw[1], rw[0], 0]])
            JL = np.zeros((3, 4))
            for c, da in enumerate(larm_dofs):
                JL[:, c] = jp[:, da] - rs @ jr[:, da]
            v_arm_L = vc[LARM_IDX]
            tL = JL.T @ (Kp_cart * eL - Kd_cart * (JL @ v_arm_L))

            eR = idR - eeR[step]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, bid_R)
            rw = d.xmat[bid_R].reshape(3, 3) @ EE_OFFSET
            rs = np.array([[0, -rw[2], rw[1]], [rw[2], 0, -rw[0]], [-rw[1], rw[0], 0]])
            JR = np.zeros((3, 4))
            for c, da in enumerate(rarm_dofs):
                JR[:, c] = jp[:, da] - rs @ jr[:, da]
            v_arm_R = vc[RARM_IDX]
            tR = JR.T @ (Kp_cart * eR - Kd_cart * (JR @ v_arm_R))

        for i, ad in enumerate(dof):
            d.qacc[ad] = apd[i]
        d.qfrc_applied[:] = 0.0
        mujoco.mj_inverse(m, d)
        d.qfrc_applied[:] = 0.0
        for ad in dof:
            d.qfrc_applied[ad] = d.qfrc_inverse[ad]
        if use_cart:
            for c in range(4):
                d.qfrc_applied[larm_dofs[c]] += tL[c]
                d.qfrc_applied[rarm_dofs[c]] += tR[c]
        mujoco.mj_step(m, d)
        t += dt

    return times, eeL, eeR


def analyze(name, times, ee, c, r, freq=0.5):
    mask = times >= 0.5
    t, e = times[mask], ee[mask]
    dist = np.linalg.norm(e - c, axis=1)
    ra_err = np.abs(dist - r)

    # 拟合圆
    yy, zz = e[:, 1], e[:, 2]
    cy, cz = c[1], c[2]
    for _ in range(8):
        di = np.sqrt((yy - cy)**2 + (zz - cz)**2)
        Rf = di.mean()
        gy = np.mean((cy - yy) * (di - Rf) / (di + 1e-12))
        gz = np.mean((cz - zz) * (di - Rf) / (di + 1e-12))
        cy -= gy * 0.3; cz -= gz * 0.3
    cfit = np.array([c[0], cy, cz])
    dfit = np.sqrt((yy - cy)**2 + (zz - cz)**2)
    Rfit = dfit.mean()
    efit = np.abs(dfit - Rfit)

    # X 方向
    xx = e[:, 0]
    x_range = xx.max() - xx.min()

    print(f"\n--- {name} ---")
    print(f"  距理想圆心: min={dist.min()*1000:.1f}mm max={dist.max()*1000:.1f}mm "
          f"mean={dist.mean()*1000:.1f}mm std={dist.std()*1000:.1f}mm")
    print(f"  径向误差(理想): max={ra_err.max()*1000:.1f}mm mean={ra_err.mean()*1000:.1f}mm")
    print(f"  X方向: range={x_range*1000:.1f}mm std={xx.std()*1000:.1f}mm")
    print(f"  拟合圆心偏移: {np.linalg.norm(cfit-c)*1000:.1f}mm  "
          f"拟合半径: {Rfit*1000:.1f}mm(理想{r*1000:.0f}mm)")
    print(f"  拟合后径向误差: max={efit.max()*1000:.1f}mm mean={efit.mean()*1000:.1f}mm "
          f"std={efit.std()*1000:.1f}mm")

    return dist, ra_err, efit


def main():
    print("=" * 60)
    print("  笛卡尔修正对比测试")

    cL = np.array([0.3485, 0.2535, 1.3016])
    cR = np.array([0.3485, -0.2535, 1.3016])
    r = 0.12

    # --- 测试1: 无笛卡尔修正 ---
    print("\n[测试1] 纯关节空间控制 (use_cart=False)")
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    qt = precompute(m, d, cL, cR, r)
    t1, eL1, eR1 = simulate(m, d, qt, cL, cR, r, use_cart=False, duration=2.5)
    d1_L, re1_L, ef1_L = analyze("左臂-无修正", t1, eL1, cL, r)
    d1_R, re1_R, ef1_R = analyze("右臂-无修正", t1, eR1, cR, r)

    # --- 测试2: 有笛卡尔修正 ---
    print("\n[测试2] 关节空间 + 笛卡尔修正 (use_cart=True, Kp_cart=8000)")
    m2 = mujoco.MjModel.from_xml_path(XML_PATH)
    d2 = mujoco.MjData(m2)
    qt2 = precompute(m2, d2, cL, cR, r)
    t2, eL2, eR2 = simulate(m2, d2, qt2, cL, cR, r, use_cart=True, duration=2.5)
    d2_L, re2_L, ef2_L = analyze("左臂-有修正", t2, eL2, cL, r)
    d2_R, re2_R, ef2_R = analyze("右臂-有修正", t2, eR2, cR, r)

    # --- 对比 ---
    print(f"\n{'='*60}")
    print(f"  对比汇总")
    print(f"{'='*60}")
    print(f"  {'':20s}  {'max_err':>10s}  {'mean_err':>10s}  {'X_range':>10s}")
    for label, re, xx in [
        ("左-无修正", re1_L, eL1[t1>=0.5][:,0].max()-eL1[t1>=0.5][:,0].min()),
        ("左-有修正", re2_L, eL2[t2>=0.5][:,0].max()-eL2[t2>=0.5][:,0].min()),
        ("右-无修正", re1_R, eR1[t1>=0.5][:,0].max()-eR1[t1>=0.5][:,0].min()),
        ("右-有修正", re2_R, eR2[t2>=0.5][:,0].max()-eR2[t2>=0.5][:,0].min()),
    ]:
        print(f"  {label:20s}  {re.max()*1000:8.1f}mm  {re.mean()*1000:8.1f}mm  "
              f"{xx*1000:8.1f}mm")

    with open("cart_compare.txt", "w", encoding="utf-8") as f:
        f.write("对比结果\n")
        for label, re, xx in [
            ("左-无修正", re1_L, eL1[t1>=0.5][:,0].max()-eL1[t1>=0.5][:,0].min()),
            ("左-有修正", re2_L, eL2[t2>=0.5][:,0].max()-eL2[t2>=0.5][:,0].min()),
            ("右-无修正", re1_R, eR1[t1>=0.5][:,0].max()-eR1[t1>=0.5][:,0].min()),
            ("右-有修正", re2_R, eR2[t2>=0.5][:,0].max()-eR2[t2>=0.5][:,0].min()),
        ]:
            f.write(f"{label}: max_err={re.max()*1000:.1f}mm mean_err={re.mean()*1000:.1f}mm X_range={xx*1000:.1f}mm\n")

    print(f"\n详细结果已保存至 cart_compare.txt")
    print(f"分析完成。")


if __name__ == '__main__':
    main()
