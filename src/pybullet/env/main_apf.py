import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pybullet.env.pointcloud_tools import load_workspace_bounds


def create_marker(position, radius, rgba):
    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
    )


def solve_ik_continuous(robot_id, end_effector_idx, target_pos, target_orn, joint_indices):
    if not p.isConnected():
        return None
    try:
        rest_q = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        if target_orn is None:
            ik = p.calculateInverseKinematics(
                robot_id,
                end_effector_idx,
                target_pos.tolist(),
                restPoses=rest_q,
            )
        else:
            ik = p.calculateInverseKinematics(
                robot_id,
                end_effector_idx,
                target_pos.tolist(),
                targetOrientation=target_orn,
                restPoses=rest_q,
            )
        return list(ik)[: len(joint_indices)]
    except Exception:
        return None


def compute_limited_joint_targets(robot_id, joint_indices, raw_targets, max_step_rad):
    if not p.isConnected():
        return None
    curr_q = np.array([p.getJointState(robot_id, j)[0] for j in joint_indices], dtype=np.float64)
    raw_q = np.asarray(raw_targets, dtype=np.float64)
    delta = np.clip(raw_q - curr_q, -max_step_rad, max_step_rad)
    return (curr_q + delta).tolist()


def hold_current_posture(robot_id, joint_indices):
    q_hold = [p.getJointState(robot_id, j)[0] for j in joint_indices]
    p.setJointMotorControlArray(
        robot_id,
        joint_indices,
        p.POSITION_CONTROL,
        targetPositions=q_hold,
        positionGains=[0.06] * len(joint_indices),
        velocityGains=[0.5] * len(joint_indices),
    )


def safe_step_sim(client_id):
    if not p.isConnected(client_id):
        return False
    try:
        p.stepSimulation()
        return True
    except Exception:
        return False


def build_tube_motion(
    start_pos,
    goal_pos,
    tube_radius=0.07,
    endpoint_clearance=0.18,
    line_angular_speed=0.35,
    swirl_angular_speed=0.80,
):
    start_pos = np.asarray(start_pos, dtype=np.float64)
    goal_pos = np.asarray(goal_pos, dtype=np.float64)
    seg = goal_pos - start_pos
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-8:
        raise ValueError('start and goal are too close; cannot build tube motion')

    seg_dir = seg / seg_len
    basis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(seg_dir, basis))) > 0.95:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    n1 = np.cross(seg_dir, basis)
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(seg_dir, n1)

    endpoint_margin = float(np.clip(endpoint_clearance / max(seg_len, 1e-9), 0.0, 0.45))
    alpha_min = endpoint_margin
    alpha_max = 1.0 - endpoint_margin

    def obstacle_pos(t_abs):
        line_phase = 0.5 * (1.0 + np.sin(line_angular_speed * t_abs))
        line_phase = float(np.clip(line_phase, alpha_min, alpha_max))
        swirl_phase = swirl_angular_speed * t_abs
        base = start_pos + line_phase * seg
        offset = tube_radius * (np.cos(swirl_phase) * n1 + np.sin(swirl_phase) * n2)
        return base + offset

    return obstacle_pos


def interpolate_by_distance(distance, near_dist, far_dist, near_value, far_value):
    if distance <= near_dist:
        return float(near_value)
    if distance >= far_dist:
        return float(far_value)
    ratio = (distance - near_dist) / (far_dist - near_dist)
    return float(near_value + ratio * (far_value - near_value))


def compute_apf_direction(
    current_pos,
    goal_pos,
    obstacle_center,
    obstacle_radius,
    influence_distance,
    k_att,
    k_rep,
):
    current_pos = np.asarray(current_pos, dtype=np.float64)
    goal_pos = np.asarray(goal_pos, dtype=np.float64)
    obstacle_center = np.asarray(obstacle_center, dtype=np.float64)

    to_goal = goal_pos - current_pos
    goal_dist = float(np.linalg.norm(to_goal))
    if goal_dist < 1e-9:
        return np.zeros(3, dtype=np.float64), np.inf

    f_att = k_att * to_goal

    away_vec = current_pos - obstacle_center
    center_dist = float(np.linalg.norm(away_vec))
    if center_dist < 1e-9:
        away_dir = -to_goal / max(goal_dist, 1e-9)
    else:
        away_dir = away_vec / center_dist

    surface_dist = center_dist - float(obstacle_radius)
    f_rep = np.zeros(3, dtype=np.float64)
    if surface_dist < influence_distance:
        safe_dist = max(surface_dist, 0.01)
        rep_gain = (1.0 / safe_dist - 1.0 / influence_distance) / (safe_dist * safe_dist)
        f_rep = k_rep * rep_gain * away_dir

    f_total = f_att + f_rep
    norm_total = float(np.linalg.norm(f_total))
    if norm_total < 1e-9:
        unit_dir = to_goal / goal_dist
    else:
        unit_dir = f_total / norm_total

    return unit_dir, surface_dist


def main():
    client_id = p.connect(p.GUI)
    if client_id < 0:
        client_id = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.loadURDF('plane.urdf')
    robot_id = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0], useFixedBase=True)
    end_effector_idx = 11
    joint_indices = list(range(7))

    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'env.yaml')
    bounds_min, bounds_max = load_workspace_bounds(config_path)

    start_pos = np.array([0.56, 0.16, 0.44], dtype=np.float64)
    goal_pos = np.array([0.20, -0.20, 0.72], dtype=np.float64)

    q_start = solve_ik_continuous(robot_id, end_effector_idx, start_pos, None, joint_indices)
    q_goal = solve_ik_continuous(robot_id, end_effector_idx, goal_pos, None, joint_indices)
    if q_start is None or q_goal is None:
        raise RuntimeError('Fixed start/goal are not IK-reachable. Please adjust start_pos/goal_pos.')

    obstacle_radius = 0.09
    obstacle_center_fn = build_tube_motion(
        start_pos,
        goal_pos,
        tube_radius=0.07,
        endpoint_clearance=0.18,
        line_angular_speed=0.35,
        swirl_angular_speed=0.80,
    )
    obstacle_center = np.clip(obstacle_center_fn(0.0), bounds_min, bounds_max)

    visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=obstacle_radius, rgbaColor=[1, 0, 0, 1])
    collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius)
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=obstacle_center.tolist(),
    )

    _ = obstacle_id
    _ = create_marker(start_pos.tolist(), radius=0.03, rgba=[0.1, 1.0, 0.1, 0.9])
    _ = create_marker(goal_pos.tolist(), radius=0.03, rgba=[1.0, 0.1, 0.1, 0.9])

    # APF + adaptive speed/step params.
    influence_distance = 0.35
    k_att = 1.8
    k_rep = 0.003

    near_clearance = 0.08
    far_clearance = 0.35

    ee_speed_near = 0.02
    ee_speed_far = 0.16

    joint_step_near = 0.004
    joint_step_far = 0.015

    sim_dt = 1.0 / 240.0
    t = 0.0
    fixed_ee_orn = None
    status_print_interval = 0.25
    last_status_print_t = -1e9

    # Move to fixed start.
    for _ in range(1500):
        if not p.isConnected(client_id):
            break
        ee_now = np.asarray(p.getLinkState(robot_id, end_effector_idx)[4], dtype=np.float64)
        if np.linalg.norm(ee_now - start_pos) <= 0.03:
            break

        q_raw = solve_ik_continuous(robot_id, end_effector_idx, start_pos, fixed_ee_orn, joint_indices)
        if q_raw is not None:
            q_cmd = compute_limited_joint_targets(robot_id, joint_indices, q_raw, max_step_rad=0.02)
            if q_cmd is not None:
                p.setJointMotorControlArray(
                    robot_id,
                    joint_indices,
                    p.POSITION_CONTROL,
                    targetPositions=q_cmd,
                    positionGains=[0.08] * 7,
                    velocityGains=[0.6] * 7,
                )

        if not safe_step_sim(client_id):
            return
        time.sleep(sim_dt)

    print('Start APF adaptive control loop...')
    while p.isConnected(client_id):
        t += sim_dt

        obstacle_center = np.clip(obstacle_center_fn(t), bounds_min, bounds_max)
        p.resetBasePositionAndOrientation(obstacle_id, obstacle_center.tolist(), [0, 0, 0, 1])

        ee_pos = np.asarray(p.getLinkState(robot_id, end_effector_idx)[4], dtype=np.float64)
        dist_to_goal = float(np.linalg.norm(goal_pos - ee_pos))
        if dist_to_goal < 0.05:
            print('Goal reached with APF controller.')
            break

        apf_dir, clearance = compute_apf_direction(
            current_pos=ee_pos,
            goal_pos=goal_pos,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            influence_distance=influence_distance,
            k_att=k_att,
            k_rep=k_rep,
        )

        ee_speed = interpolate_by_distance(
            distance=clearance,
            near_dist=near_clearance,
            far_dist=far_clearance,
            near_value=ee_speed_near,
            far_value=ee_speed_far,
        )
        ee_step = min(ee_speed * sim_dt, dist_to_goal)

        max_step_rad = interpolate_by_distance(
            distance=clearance,
            near_dist=near_clearance,
            far_dist=far_clearance,
            near_value=joint_step_near,
            far_value=joint_step_far,
        )

        ee_target = ee_pos + apf_dir * ee_step
        q_raw = solve_ik_continuous(robot_id, end_effector_idx, ee_target, fixed_ee_orn, joint_indices)
        if q_raw is None:
            hold_current_posture(robot_id, joint_indices)
        else:
            q_cmd = compute_limited_joint_targets(
                robot_id,
                joint_indices,
                q_raw,
                max_step_rad=max_step_rad,
            )
            if q_cmd is None:
                hold_current_posture(robot_id, joint_indices)
            else:
                p.setJointMotorControlArray(
                    robot_id,
                    joint_indices,
                    p.POSITION_CONTROL,
                    targetPositions=q_cmd,
                    positionGains=[0.08] * 7,
                    velocityGains=[0.6] * 7,
                )

        if (t - last_status_print_t) >= status_print_interval:
            print(
                f't={t:.2f}s, goal_dist={dist_to_goal:.3f}, clearance={clearance:.3f}, '
                f'ee_speed={ee_speed:.3f} m/s, ee_step={ee_step:.4f} m, joint_step={max_step_rad:.4f} rad'
            )
            last_status_print_t = t

        if not safe_step_sim(client_id):
            break
        time.sleep(sim_dt)

    if p.isConnected(client_id):
        p.disconnect(client_id)


if __name__ == '__main__':
    main()
