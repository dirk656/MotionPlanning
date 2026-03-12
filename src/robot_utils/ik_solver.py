import numpy as np
from robot_utils.fk_solver import precompute_ik_cache, fk_for_ik


def numerical_ik(links, joints, target_pos, base_pos=(0, 0, 0),
                 joint_limits=None, q_init=None,
                 max_iter=200, tol=1e-3,
                 ee_link=None, _ik_cache=None):
    """
    数值迭代逆运动学（解析几何雅可比 + 阻尼最小二乘），仅求解位置（3-DOF 目标）。
    每次迭代只需 1 次 FK（而非 n_dof+1 次有限差分）。
    """
    target_pos = np.asarray(target_pos, dtype=float)

    revolute_joints = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    n_dof = len(revolute_joints)

    if ee_link is None:
        ee_link = revolute_joints[-1]['child'] if revolute_joints else joints[-1]['child']

    if joint_limits is not None:
        joint_limits = np.asarray(joint_limits)
    else:
        joint_limits = np.full((n_dof, 2), [-np.pi, np.pi])

    if q_init is not None:
        q = np.array(q_init, dtype=float).copy()
    else:
        q = (joint_limits[:, 0] + joint_limits[:, 1]) * 0.5

    # 预计算静态数据（可由 solve_ik_multi_start 传入以避免重复计算）
    if _ik_cache is None:
        _ik_cache = precompute_ik_cache(joints)
    axes = _ik_cache['axes']

    base_T = np.eye(4)
    base_T[:3, 3] = np.asarray(base_pos, dtype=float)
    I3 = np.eye(3)

    for _ in range(max_iter):
        # 一次 FK 同时得到末端位置和各关节世界坐标系
        ee_pos, joint_frames = fk_for_ik(joints, q, base_T, _ik_cache, ee_link)

        error = target_pos - ee_pos
        err_norm = np.linalg.norm(error)
        if err_norm < tol:
            return q, True

        # 解析几何雅可比 (3 × n_dof): J_i = z_i × (p_ee − p_i)
        J = np.zeros((3, n_dof))
        for i in range(n_dof):
            Tf = joint_frames[i]
            z_i = Tf[:3, :3] @ axes[i]
            p_i = Tf[:3, 3]
            J[:, i] = np.cross(z_i, ee_pos - p_i)

        # 阻尼最小二乘（DLS），全步更新
        damping = 1e-2
        JJT = J @ J.T + damping * I3
        dq = J.T @ np.linalg.solve(JJT, error)

        q = q + dq
        q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])

    return q, False


def solve_ik_multi_start(links, joints, target_pos, base_pos=(0, 0, 0),
                         joint_limits=None, n_starts=20,
                         max_iter=200, tol=1e-3, ee_link=None,
                         collision_checker=None):
    """
    多初始值 IK 求解，返回误差最小的结果。
    collision_checker: 可选，callable(q) -> bool，返回 True 表示无碰撞。
                       有碰撞检查时，优先返回无碰撞的精确解。
    """
    revolute_joints = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    n_dof = len(revolute_joints)

    if joint_limits is not None:
        joint_limits = np.array(joint_limits)
    else:
        joint_limits = np.array([[-np.pi, np.pi]] * n_dof)

    if ee_link is None:
        ee_link = revolute_joints[-1]['child'] if revolute_joints else joints[-1]['child']

    # 预计算 IK 缓存，所有 start 共享
    _ik_cache = precompute_ik_cache(joints)

    base_T = np.eye(4)
    base_T[:3, 3] = np.asarray(base_pos, dtype=float)

    best_q = None
    best_err = np.inf
    best_collision_free_q = None
    best_collision_free_err = np.inf

    for i in range(n_starts):
        if i == 0:
            q0 = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
        else:
            q0 = joint_limits[:, 0] + np.random.rand(n_dof) * (joint_limits[:, 1] - joint_limits[:, 0])

        q, success = numerical_ik(
            links, joints, target_pos, base_pos=base_pos,
            joint_limits=joint_limits, q_init=q0,
            max_iter=max_iter, tol=tol, ee_link=ee_link,
            _ik_cache=_ik_cache
        )
        if q is not None:
            ee_pos, _ = fk_for_ik(joints, q, base_T, _ik_cache, ee_link)
            err = np.linalg.norm(np.asarray(target_pos) - ee_pos)

            if err < best_err:
                best_err = err
                best_q = q.copy()

            if collision_checker is not None and success:
                if collision_checker(q):
                    if err < best_collision_free_err:
                        best_collision_free_err = err
                        best_collision_free_q = q.copy()
                    return best_collision_free_q, True
            elif collision_checker is None and success:
                return best_q, True

    if best_collision_free_q is not None and best_collision_free_err < tol:
        return best_collision_free_q, True

    return best_q, (best_err < tol)
