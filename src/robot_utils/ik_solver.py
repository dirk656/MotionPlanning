import numpy as np
from robot_utils.fk_solver import compute_fk
from robot_utils.rotation_matrix import rotation_matrix_axis_angle


def numerical_ik(links, joints, target_pos, base_pos=(0, 0, 0),
                 joint_limits=None, q_init=None,
                 max_iter=1000, tol=1e-3, alpha=0.5,
                 ee_link=None):
    """
    数值迭代逆运动学（雅可比伪逆法），仅求解位置（3-DOF 目标）。

    Args:
        links, joints: parse_urdf 返回的结构
        target_pos: (3,) 目标末端位置
        base_pos: 机械臂基座位置
        joint_limits: (n_dof, 2) 关节上下限，None 则不约束
        q_init: 初始关节角，None 则随机
        max_iter: 最大迭代次数
        tol: 位置误差容忍度
        alpha: 步长因子
        ee_link: 末端执行器 link 名称，None 则取运动链最后一个 link

    Returns:
        q: (n_dof,) 关节角度，求解失败返回 None
        success: bool
    """
    target_pos = np.array(target_pos, dtype=float)

    # 提取可动关节
    revolute_joints = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    n_dof = len(revolute_joints)
    joint_names = [j['name'] for j in revolute_joints]

    if ee_link is None:
        ee_link = revolute_joints[-1]['child'] if revolute_joints else joints[-1]['child']

    if joint_limits is not None:
        joint_limits = np.array(joint_limits)
    else:
        joint_limits = np.array([[-np.pi, np.pi]] * n_dof)

    if q_init is not None:
        q = np.array(q_init, dtype=float).copy()
    else:
        q = (joint_limits[:, 0] + joint_limits[:, 1]) / 2

    delta = 1e-4  # 有限差分步长

    for _ in range(max_iter):
        # 当前 FK
        ja = {joint_names[i]: q[i] for i in range(n_dof)}
        _, link_T, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos)
        ee_T = link_T.get(ee_link)
        if ee_T is None:
            return None, False
        ee_pos = ee_T[:3, 3]

        error = target_pos - ee_pos
        if np.linalg.norm(error) < tol:
            return q, True

        # 数值雅可比 (3 x n_dof)
        J = np.zeros((3, n_dof))
        for i in range(n_dof):
            q_perturb = q.copy()
            q_perturb[i] += delta
            ja_p = {joint_names[k]: q_perturb[k] for k in range(n_dof)}
            _, link_T_p, _ = compute_fk(links, joints, joint_angles=ja_p, base_pos=base_pos)
            ee_pos_p = link_T_p[ee_link][:3, 3]
            J[:, i] = (ee_pos_p - ee_pos) / delta

        # 阻尼最小二乘（DLS）
        damping = 1e-3
        JJT = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        q = q + alpha * dq
        # 限位
        q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])

    return q, False


def solve_ik_multi_start(links, joints, target_pos, base_pos=(0, 0, 0),
                         joint_limits=None, n_starts=20,
                         max_iter=1000, tol=1e-3, ee_link=None):
    """
    多初始值 IK 求解，返回误差最小的结果。
    """
    revolute_joints = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    n_dof = len(revolute_joints)

    if joint_limits is not None:
        joint_limits = np.array(joint_limits)
    else:
        joint_limits = np.array([[-np.pi, np.pi]] * n_dof)

    best_q = None
    best_err = np.inf

    for i in range(n_starts):
        if i == 0:
            q0 = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
        else:
            q0 = joint_limits[:, 0] + np.random.rand(n_dof) * (joint_limits[:, 1] - joint_limits[:, 0])

        q, success = numerical_ik(
            links, joints, target_pos, base_pos=base_pos,
            joint_limits=joint_limits, q_init=q0,
            max_iter=max_iter, tol=tol, ee_link=ee_link
        )
        if q is not None:
            ja = {revolute_joints[k]['name']: q[k] for k in range(n_dof)}
            _, link_T, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos)
            if ee_link is None:
                ee_link_name = revolute_joints[-1]['child']
            else:
                ee_link_name = ee_link
            ee_pos = link_T[ee_link_name][:3, 3]
            err = np.linalg.norm(np.array(target_pos) - ee_pos)
            if err < best_err:
                best_err = err
                best_q = q.copy()
            if success:
                return best_q, True

    return best_q, (best_err < tol)
