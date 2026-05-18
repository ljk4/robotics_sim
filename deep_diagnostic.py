# -*- coding: utf-8 -*-
"""深度诊断: 排查圆度问题的根源"""
import numpy as np
import mujoco
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    for it in range(max_iter):
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
    return q, it + 1, np.linalg.norm(e)


def dual_ik(m, d, tL, tR, qi, max_iter=500):
    q = np.copy(qi)
    q, itL, eL = ik_one(m, d, tL, "left",  q, max_iter)
    q, itR, eR = ik_one(m, d, tR, "right", q, max_iter)
    return q, itL, itR, eL, eR


def check_joint_limits(q, m):
    """检查哪些关节处于限位 (距离限位 < 0.005 rad)"""
    at_limit = []
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = m.jnt_range[jid]
        if abs(q[i] - lo) < 0.005:
            at_limit.append(f"{name}=LO({lo:.3f})")
        elif abs(q[i] - hi) < 0.005:
            at_limit.append(f"{name}=HI({hi:.3f})")
    return at_limit


def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    cL = np.array([0.3485, 0.2535, 1.3016])
    cR = np.array([0.3485, -0.2535, 1.3016])
    r = 0.12
    nw = 172

    print("=" * 70)
    print("  深度诊断: 圆度问题根源排查")
    print("=" * 70)

    # === 诊断1: IK 在不同 damping 下的表现 ===
    print("\n" + "=" * 70)
    print("  诊断1: IK damping 参数对比 (测试 k=0, 90, 180度三个点)")
    print("=" * 70)

    for damping in [0.1, 0.05, 0.01, 0.001]:
        print(f"\n  damping={damping}:")
        for k_deg in [0, 45, 90, 135, 180]:
            ang = np.radians(k_deg)
            tL = cL + np.array([0., r * np.cos(ang), r * np.sin(ang)])
            tR = cR + np.array([0., r * np.cos(ang + np.pi), r * np.sin(ang + np.pi)])
            q_init = np.zeros(m.nv)
            q, itL, itR, eL, eR = dual_ik(m, d, tL, tR, q_init, max_iter=2000)
            ll = check_joint_limits(q, m)
            rL = np.linalg.norm(get_ee_pos(m, d, "left") - cL)
            rR = np.linalg.norm(get_ee_pos(m, d, "right") - cR)
            rerrL = abs(rL - r) * 1000
            rerrR = abs(rR - r) * 1000
            lim_str = " *** 限位:" + ",".join(ll) if ll else ""
            print(f"    k={k_deg:3d}deg  itL={itL:4d} itR={itR:4d}  "
                  f"pos_errL={eL*1000:.2f}mm  pos_errR={eR*1000:.2f}mm  "
                  f"r_errL={rerrL:.1f}mm  r_errR={rerrR:.1f}mm{lim_str}")

    # === 诊断2: 全172点 trajectory 分析 (damping=0.1, standard) ===
    print("\n" + "=" * 70)
    print("  诊断2: 172 waypoint 逐点分析 (damping=0.1, max_iter=300)")
    print("=" * 70)

    # Build trajectory (standard method)
    qc, iLc, iRc, eLc, eRc = dual_ik(m, d, cL, cR, np.zeros(m.nv), max_iter=1000)
    print(f"  圆心 IK: itL={iLc} itR={iRc} errL={eLc*1000:.2f}mm errR={eRc*1000:.2f}mm")

    qt, qp = [], qc
    ik_stats = []
    for k in range(nw):
        ang = 2 * np.pi * k / nw
        tL = cL + np.array([0., r * np.cos(ang), r * np.sin(ang)])
        tR = cR + np.array([0., r * np.cos(ang + np.pi), r * np.sin(ang + np.pi)])
        q, itL, itR, eL, eR = dual_ik(m, d, tL, tR, qp, max_iter=300)
        qt.append(np.copy(q))
        set_q(m, d, q)
        eeL = get_ee_pos(m, d, "left")
        eeR = get_ee_pos(m, d, "right")
        rL = np.linalg.norm(eeL - cL)
        rR = np.linalg.norm(eeR - cR)
        xL, xR = eeL[0], eeR[0]
        lims = check_joint_limits(q, m)
        ik_stats.append({
            'k': k, 'ang_deg': np.degrees(ang),
            'itL': itL, 'itR': itR,
            'pos_errL_mm': eL * 1000, 'pos_errR_mm': eR * 1000,
            'rad_errL_mm': abs(rL - r) * 1000, 'rad_errR_mm': abs(rR - r) * 1000,
            'rL_mm': rL * 1000, 'rR_mm': rR * 1000,
            'xL_mm': xL * 1000, 'xR_mm': xR * 1000,
            'limits': lims,
        })
        qp = q

    stats = {k: np.array([s[k] for s in ik_stats]) for k in ['rad_errL_mm', 'rad_errR_mm', 'pos_errL_mm', 'pos_errR_mm', 'xL_mm', 'xR_mm']}

    print(f"\n  === 汇总统计 ===")
    print(f"  左臂 pos_err: max={stats['pos_errL_mm'].max():.2f}mm mean={stats['pos_errL_mm'].mean():.2f}mm")
    print(f"  右臂 pos_err: max={stats['pos_errR_mm'].max():.2f}mm mean={stats['pos_errR_mm'].mean():.2f}mm")
    print(f"  左臂 rad_err: max={stats['rad_errL_mm'].max():.2f}mm mean={stats['rad_errL_mm'].mean():.2f}mm")
    print(f"  右臂 rad_err: max={stats['rad_errR_mm'].max():.2f}mm mean={stats['rad_errR_mm'].mean():.2f}mm")
    print(f"  左臂 X:      min={stats['xL_mm'].min():.1f}mm max={stats['xL_mm'].max():.1f}mm range={stats['xL_mm'].max()-stats['xL_mm'].min():.1f}mm")
    print(f"  右臂 X:      min={stats['xR_mm'].min():.1f}mm max={stats['xR_mm'].max():.1f}mm range={stats['xR_mm'].max()-stats['xR_mm'].min():.1f}mm")

    # 找出误差最大的 waypoint
    print(f"\n  === 左臂径向误差最大的10个waypoint ===")
    top10L = np.argsort(stats['rad_errL_mm'])[::-1][:10]
    for rank, idx in enumerate(top10L):
        s = ik_stats[idx]
        lim_str = " 限位:" + ",".join(s['limits']) if s['limits'] else ""
        print(f"    #{rank+1}: k={s['k']:3d} {s['ang_deg']:6.1f}deg  "
              f"rad_err={s['rad_errL_mm']:.1f}mm  pos_err={s['pos_errL_mm']:.2f}mm  "
              f"it={s['itL']:4d}  r={s['rL_mm']:.1f}mm  x={s['xL_mm']:.1f}mm{lim_str}")

    print(f"\n  === 右臂径向误差最大的10个waypoint ===")
    top10R = np.argsort(stats['rad_errR_mm'])[::-1][:10]
    for rank, idx in enumerate(top10R):
        s = ik_stats[idx]
        lim_str = " 限位:" + ",".join(s['limits']) if s['limits'] else ""
        print(f"    #{rank+1}: k={s['k']:3d} {s['ang_deg']:6.1f}deg  "
              f"rad_err={s['rad_errR_mm']:.1f}mm  pos_err={s['pos_errR_mm']:.2f}mm  "
              f"it={s['itR']:4d}  r={s['rR_mm']:.1f}mm  x={s['xR_mm']:.1f}mm{lim_str}")

    # 找出有 joint limit 的 waypoint
    print(f"\n  === 有关节限位的waypoint ===")
    lim_count = 0
    for s in ik_stats:
        if s['limits']:
            lim_count += 1
            if lim_count <= 10:
                print(f"    k={s['k']:3d} {s['ang_deg']:6.1f}deg  {', '.join(s['limits'])}  "
                      f"rad_errL={s['rad_errL_mm']:.1f}mm  rad_errR={s['rad_errR_mm']:.1f}mm")
    print(f"  共 {lim_count}/{nw} 个 waypoint 触碰到关节限位")

    # 闭合点分析
    qt_arr = np.array(qt)
    dq_wrap = np.linalg.norm(qt_arr[0] - qt_arr[-1])
    print(f"\n  === 闭合点分析 ===")
    print(f"  |q[0] - q[-1]| = {dq_wrap:.4f} rad")
    if dq_wrap > 0.01:
        for i, label in enumerate(JLAB):
            diff = abs(qt_arr[0, i] - qt_arr[-1, i])
            flag = " ***" if diff > 0.01 else ""
            print(f"    {label}: q[0]={qt_arr[0,i]:.4f}  q[-1]={qt_arr[-1,i]:.4f}  diff={diff:.4f}{flag}")

    # === 诊断3: 关节空间插值误差 ===
    print("\n" + "=" * 70)
    print("  诊断3: 关节空间线性插值误差 (每段5个插值点)")
    print("=" * 70)
    interp_err_L, interp_err_R = [], []
    for k in range(nw):
        q0 = qt_arr[k]
        q1 = qt_arr[(k + 1) % nw]
        ang0 = 2 * np.pi * k / nw
        ang1 = 2 * np.pi * (k + 1) / nw
        for frac in np.linspace(0.1, 0.9, 5):
            q_mid = (1 - frac) * q0 + frac * q1
            set_q(m, d, q_mid)
            eeL = get_ee_pos(m, d, "left")
            eeR = get_ee_pos(m, d, "right")
            ang_mid = (1 - frac) * ang0 + frac * ang1
            idealL = cL + np.array([0., r * np.cos(ang_mid), r * np.sin(ang_mid)])
            idealR = cR + np.array([0., r * np.cos(ang_mid + np.pi), r * np.sin(ang_mid + np.pi)])
            interp_err_L.append(np.linalg.norm(eeL - idealL))
            interp_err_R.append(np.linalg.norm(eeR - idealR))
    iel = np.array(interp_err_L) * 1000
    ier = np.array(interp_err_R) * 1000
    print(f"  左臂插值误差: max={iel.max():.1f}mm mean={iel.mean():.1f}mm std={iel.std():.1f}mm")
    print(f"  右臂插值误差: max={ier.max():.1f}mm mean={ier.mean():.1f}mm std={ier.std():.1f}mm")

    # 找出插值误差最大的段
    seg_err_L = [np.mean(iel[k*5:(k+1)*5]) for k in range(nw)]
    seg_err_R = [np.mean(ier[k*5:(k+1)*5]) for k in range(nw)]
    worst_seg_L = np.argmax(seg_err_L)
    worst_seg_R = np.argmax(seg_err_R)
    print(f"  左臂最大插值误差段: wp[{worst_seg_L}]->[{worst_seg_L+1}] mean={seg_err_L[worst_seg_L]:.1f}mm")
    print(f"  右臂最大插值误差段: wp[{worst_seg_R}]->[{worst_seg_R+1}] mean={seg_err_R[worst_seg_R]:.1f}mm")

    # === 诊断4: 雅可比条件数检查 ===
    print("\n" + "=" * 70)
    print("  诊断4: 雅可比条件数 (可操作性)")
    print("=" * 70)
    for k_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        ang = np.radians(k_deg)
        tL = cL + np.array([0., r * np.cos(ang), r * np.sin(ang)])
        tR = cR + np.array([0., r * np.cos(ang + np.pi), r * np.sin(ang + np.pi)])
        q, _, _, _, _ = dual_ik(m, d, tL, tR, np.zeros(m.nv), max_iter=1000)
        set_q(m, d, q)
        for side, color in [("left", "L"), ("right", "R")]:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, bid)
            rw = d.xmat[bid].reshape(3, 3) @ EE_OFFSET
            rsk = np.array([[0, -rw[2], rw[1]], [rw[2], 0, -rw[0]], [-rw[1], rw[0], 0]])
            jidx = LARM_IDX if side == "left" else RARM_IDX
            J = np.zeros((3, len(jidx)))
            for c, idx in enumerate(jidx):
                da = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[idx])]
                J[:, c] = jp[:, da] - rsk @ jr[:, da]
            s = np.linalg.svd(J, compute_uv=False)
            cn = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
            print(f"  {color} {k_deg:3d}deg: svd={np.round(s,3)} cond={cn:.1f}")

    print("\n" + "=" * 70)
    print("  诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
