import pybullet as p
import pybullet_data
import time
import numpy as np
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pybullet.env.pointcloud_tools import (
    load_workspace_bounds,
    depth_buffer_to_points_world,
    filter_points_by_bounds,
    voxelize_points,
    AsyncPointCloudNode,
)
from src.robot_utils.time_based_rrt import (
    HeuristicTimeRRTPlanner,
    OnlineHeuristicPlanningManager,
    AsyncPlanningNode,
    PyBulletArmCollisionChecker,
    SphereObstacle,
)
from src.robot_utils.predictor import HeuristicRegionPredictor
from src.pybullet.env.path_tools import generate_pos

def get_current_pointcloud(bounds_min, bounds_max, proj_matrix, voxel_size=0.03, max_voxels=1200):
    cam_target = [0.25, 0.0, 0.25]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target, distance=1.3, yaw=45, pitch=-35, roll=0, upAxisIndex=2
    )
    # Get camera image and depth info
    _, _, _, depth_buffer, _ = p.getCameraImage(
        width=320, height=240, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    depth_buffer = np.asarray(depth_buffer, dtype=np.float64)
    points = depth_buffer_to_points_world(depth_buffer, view_matrix, proj_matrix)
    points = filter_points_by_bounds(points, bounds_min, bounds_max)
    voxel_points, voxel_aabb = voxelize_points(points, voxel_size=voxel_size, max_voxels=max_voxels)
    return voxel_points, voxel_aabb


def capture_depth_sensor_frame(proj_matrix):
    cam_target = [0.25, 0.0, 0.25]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target, distance=1.3, yaw=45, pitch=-35, roll=0, upAxisIndex=2
    )
    _, _, _, depth_buffer, _ = p.getCameraImage(
        width=320,
        height=240,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    depth_buffer = np.asarray(depth_buffer, dtype=np.float64)
    return depth_buffer, np.asarray(view_matrix, dtype=np.float64)


def create_marker(position, radius, rgba):
    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
    )


def compute_limited_joint_targets(robot_id, joint_indices, raw_targets, max_step_rad=0.02):
    if not p.isConnected():
        return None
    curr_q = np.array([p.getJointState(robot_id, j)[0] for j in joint_indices], dtype=np.float64)
    raw_q = np.asarray(raw_targets, dtype=np.float64)
    delta = np.clip(raw_q - curr_q, -max_step_rad, max_step_rad)
    return (curr_q + delta).tolist()


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


def hold_current_posture(robot_id, joint_indices):
    if not p.isConnected():
        return
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


def update_path_debug_lines(path, prev_debug_line_ids):
    if not p.isConnected():
        return []

    for line_id in prev_debug_line_ids:
        try:
            p.removeUserDebugItem(line_id)
        except Exception:
            return []

    new_ids = []
    if path is None or len(path) < 2:
        return new_ids

    for idx in range(len(path) - 1):
        p0 = np.asarray(path[idx][0], dtype=np.float64)
        p1 = np.asarray(path[idx + 1][0], dtype=np.float64)
        try:
            line_id = p.addUserDebugLine(
                p0.tolist(),
                p1.tolist(),
                lineColorRGB=[0.0, 0.9, 1.0],
                lineWidth=2.5,
                lifeTime=0,
            )
            new_ids.append(line_id)
        except Exception:
            return new_ids
    return new_ids


def build_tube_motion(start_pos, goal_pos, tube_radius=0.06, endpoint_clearance=0.16):
    start_pos = np.asarray(start_pos, dtype=np.float64)
    goal_pos = np.asarray(goal_pos, dtype=np.float64)
    seg = goal_pos - start_pos
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-8:
        raise ValueError("start and goal are too close; cannot build tube motion")

    seg_dir = seg / seg_len
    basis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(seg_dir, basis))) > 0.95:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    n1 = np.cross(seg_dir, basis)
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(seg_dir, n1)

    # Keep obstacle away from start/goal neighborhoods.
    # alpha=0 is start, alpha=1 is goal, so we clamp alpha to a safe mid segment.
    endpoint_margin = float(np.clip(endpoint_clearance / max(seg_len, 1e-9), 0.0, 0.45))
    alpha_min = endpoint_margin
    alpha_max = 1.0 - endpoint_margin

    def obstacle_pos(t_abs):
        line_phase = 0.5 * (1.0 + np.sin(0.9 * t_abs))
        line_phase = float(np.clip(line_phase, alpha_min, alpha_max))
        swirl_phase = 2.3 * t_abs
        base = start_pos + line_phase * seg
        offset = tube_radius * (
            np.cos(swirl_phase) * n1 + np.sin(swirl_phase) * n2
        )
        return base + offset

    return obstacle_pos


def sample_reachable_start_goal(robot_id, end_effector_idx, joint_indices, max_tries=60):
    for _ in range(max_tries):
        start_pos, goal_pos = generate_pos(
            max_radius=0.62,
            min_start_goal_distance=0.55,
            min_robot_distance_xy=0.46,
            min_robot_distance_3d=0.56,
            min_line_robot_distance_3d=0.44,
            z_range=(0.28, 0.88),
            max_attempts=800,
        )

        q_s = solve_ik_continuous(robot_id, end_effector_idx, start_pos, None, joint_indices)
        q_g = solve_ik_continuous(robot_id, end_effector_idx, goal_pos, None, joint_indices)
        if q_s is None or q_g is None:
            continue

        old_states = [p.getJointState(robot_id, j)[0] for j in joint_indices]
        for i, j in enumerate(joint_indices):
            p.resetJointState(robot_id, j, float(q_s[i]))
        ee_s = np.asarray(p.getLinkState(robot_id, end_effector_idx)[4], dtype=np.float64)
        err_s = float(np.linalg.norm(ee_s - start_pos))

        for i, j in enumerate(joint_indices):
            p.resetJointState(robot_id, j, float(q_g[i]))
        ee_g = np.asarray(p.getLinkState(robot_id, end_effector_idx)[4], dtype=np.float64)
        err_g = float(np.linalg.norm(ee_g - goal_pos))

        for i, j in enumerate(joint_indices):
            p.resetJointState(robot_id, j, float(old_states[i]))

        if err_s <= 0.05 and err_g <= 0.05:
            return start_pos, goal_pos

    raise RuntimeError("Failed to sample IK-reachable start/goal pair")

def main():
    # 1. Initialize PyBullet
    client_id = p.connect(p.GUI)
    if client_id < 0:
        client_id = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # 2. Load scene and models
    planeId = p.loadURDF("plane.urdf")
    # Using Franka robot as in generate_pointcloud.py
    robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    end_effector_idx = 11  # Franka end effector
    num_joints = p.getNumJoints(robotId)
    joints_indices = list(range(7)) # Control the first 7 joints

    # 3. Load configuration and model predictor
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'env.yaml')

    bounds_min, bounds_max = load_workspace_bounds(config_path)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=320/240.0, nearVal=0.05, farVal=3.0)

    print("Loading PointNet Predictor...")
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'model_training', 'pointnet2_3d', 'checkpoints', 'best_pointnet2_3d')

    try:
        predictor = HeuristicRegionPredictor(
            model_root=model_path,
            env_config_path=config_path,
            device='cpu',
        )
    except Exception as e:
        print(f"Failed to load predictor: {e}")
        predictor = None

    # 4. RRT Planner Initialization
    obstacle_radius = 0.09
    obstacle_center_cache = None

    def get_dynamic_obstacles(t_abs):
        nonlocal obstacle_center_cache
        _ = t_abs
        if obstacle_center_cache is None:
            return []
        return [SphereObstacle(center=np.asarray(obstacle_center_cache, dtype=np.float64), radius=obstacle_radius)]

    rng = np.random.default_rng(42)
    planner = HeuristicTimeRRTPlanner(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        speed=0.5,
        step_len=0.035,
        goal_tolerance=0.05,
        goal_bias=0.25,
        heuristic_bias=0.55,
        max_iters=260,
        time_horizon=10.0,
        collision_margin=0.05,
    )

    # 5. Define Start, Goal, Control Loop
    use_fixed_start_goal = True
    fixed_start_pos = np.array([0.56, 0.16, 0.44], dtype=np.float64)
    fixed_goal_pos = np.array([0.40, -0.40, 0.72], dtype=np.float64)

    if use_fixed_start_goal:
        start_pos = fixed_start_pos
        goal_pos = fixed_goal_pos

        q_s = solve_ik_continuous(robotId, end_effector_idx, start_pos, None, joints_indices)
        q_g = solve_ik_continuous(robotId, end_effector_idx, goal_pos, None, joints_indices)
        if q_s is None or q_g is None:
            raise RuntimeError(
                "Fixed start/goal are not IK-reachable. Please adjust fixed_start_pos/fixed_goal_pos."
            )
    else:
        start_pos, goal_pos = sample_reachable_start_goal(
            robotId,
            end_effector_idx,
            joints_indices,
        )

    # Obstacle moves in a tube around the start-goal segment.
    obstacle_center_fn = build_tube_motion(
        start_pos,
        goal_pos,
        tube_radius=0.07,
        endpoint_clearance=0.18,
    )
    obs_init = np.clip(obstacle_center_fn(0.0), bounds_min, bounds_max)
    obstacle_center_cache = np.asarray(obs_init, dtype=np.float64)
    visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=obstacle_radius, rgbaColor=[1, 0, 0, 1])
    collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius)
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=obs_init.tolist(),
    )

    t = 0.0
    sim_dt = 1.0 / 240.0
    replan_min_interval = 0.015
    periodic_online_replan_interval = 0.2
    pointcloud_refresh_interval = 0.15
    last_replan_t = -1e9
    last_periodic_replan_t = -1e9
    last_pointcloud_t = -1e9
    waypoint_reach_tol = 0.03
    active_waypoint = None
    active_waypoint_last_progress_t = -1e9
    active_waypoint_best_dist = 1e9
    ik_fail_streak = 0
    waypoint_stall_timeout = 0.7
    max_ik_fail_streak = 25
    path_debug_line_ids = []
    cached_pc = None
    cached_pc_aabb = None

    start_marker_id = create_marker(start_pos.tolist(), radius=0.03, rgba=[0.1, 1.0, 0.1, 0.9])
    goal_marker_id = create_marker(goal_pos.tolist(), radius=0.03, rgba=[1.0, 0.1, 0.1, 0.9])
    fixed_ee_orn = None

    arm_collision_checker = PyBulletArmCollisionChecker(
        robot_id=robotId,
        end_effector_idx=end_effector_idx,
        controlled_joint_indices=joints_indices,
        obstacle_body_ids=[obstacle_id],
        safety_distance=0.015,
    )

    planning_manager = OnlineHeuristicPlanningManager(
        planner=planner,
        predictor=predictor,
        dynamic_obstacle_fn=get_dynamic_obstacles,
        arm_collision_checker=None,
        neighbor_radius=0.1,
        max_trial_attempts=3,
        incremental_iter_budget=80,
        tree_resume_pos_tol=0.08,
        tree_resume_time_window=0.35,
        heuristic_update_interval=0.15,
        rng=rng,
    )
    planning_node = AsyncPlanningNode(
        planning_manager=planning_manager,
        iter_budget=80,
        visualize=False,
    )
    planning_node.start()

    pointcloud_node = AsyncPointCloudNode()
    pointcloud_node.start()

    shared_path = []
    shared_path_status = {}
    shared_path_version = 0
    pointcloud_version = 0
    
    # Pre-move robot to start_pos until close enough.
    reached_start = False
    for _ in range(1200):
        if not p.isConnected(client_id):
            break
        ee_now = np.asarray(p.getLinkState(robotId, end_effector_idx)[4], dtype=np.float64)
        if np.linalg.norm(ee_now - start_pos) <= 0.03:
            reached_start = True
            break
        q_raw = solve_ik_continuous(robotId, end_effector_idx, start_pos, fixed_ee_orn, joints_indices)
        if q_raw is None:
            continue
        q_cmd = compute_limited_joint_targets(robotId, joints_indices, q_raw, max_step_rad=0.02)
        if q_cmd is None:
            continue
        p.setJointMotorControlArray(
            robotId,
            joints_indices,
            p.POSITION_CONTROL,
            targetPositions=q_cmd,
            positionGains=[0.08] * 7,
            velocityGains=[0.6] * 7,
        )
        if not safe_step_sim(client_id):
            print("Physics server disconnected during init; stopping.")
            break
        time.sleep(sim_dt)

    if not p.isConnected(client_id):
        planning_node.stop()
        planning_node.join(timeout=1.0)
        pointcloud_node.stop()
        pointcloud_node.join(timeout=1.0)
        return

    ee_after_init = np.asarray(p.getLinkState(robotId, end_effector_idx)[4], dtype=np.float64)
    print(f"Init EE-start error: {np.linalg.norm(ee_after_init - start_pos):.4f} m")
    if not reached_start:
        print("Warning: could not fully reach start before planning; continuing with best-effort state.")
    
    print("Start Control Loop...")
    while p.isConnected(client_id):
        t += sim_dt
        
        # update dynamic obstacle
        obs_pos = np.clip(obstacle_center_fn(t), bounds_min, bounds_max)
        obstacle_center_cache = np.asarray(obs_pos, dtype=np.float64)
        if p.isConnected(client_id):
            p.resetBasePositionAndOrientation(obstacle_id, obs_pos.tolist(), [0, 0, 0, 1])
        else:
            break
        
        # current EE pos
        state = p.getLinkState(robotId, end_effector_idx)
        current_ee_pos = np.array(state[4]) # world link position
        
        can_replan_now = (t - last_replan_t) >= replan_min_interval
        periodic_due = (t - last_periodic_replan_t) >= periodic_online_replan_interval
        need_replan = (active_waypoint is None and len(shared_path) == 0) or periodic_due

        if can_replan_now and need_replan:
            print(f"Queue async replanning request at t={t:.2f}...")
            last_replan_t = t
            if periodic_due:
                last_periodic_replan_t = t
            try:
                if (t - last_pointcloud_t) >= pointcloud_refresh_interval:
                    depth_buffer, view_matrix = capture_depth_sensor_frame(proj_matrix)
                    pointcloud_node.submit(
                        depth_buffer=depth_buffer,
                        view_matrix=view_matrix,
                        proj_matrix=np.asarray(proj_matrix, dtype=np.float64),
                        bounds_min=bounds_min,
                        bounds_max=bounds_max,
                        voxel_size=0.03,
                        max_voxels=1200,
                    )
                    last_pointcloud_t = t
                planning_node.submit(
                    point_cloud=cached_pc,
                    static_obs_aabb=cached_pc_aabb,
                    current_pos=current_ee_pos,
                    goal_pos=goal_pos,
                    current_t=t,
                )
            except Exception as replanning_exc:
                if not p.isConnected(client_id):
                    print("Physics server disconnected during replanning; stopping.")
                    break
                print(f"Replanning skipped due to transient simulation error: {replanning_exc}")
                continue

        # Pull latest point cloud from worker thread with atomic snapshot.
        new_pc_version, new_pc, new_pc_aabb = pointcloud_node.result_store.get_if_newer(pointcloud_version)
        if new_pc is not None:
            pointcloud_version = new_pc_version
            cached_pc = new_pc
            cached_pc_aabb = new_pc_aabb

        # Read latest planner result atomically from background thread.
        new_version, new_path, status = planning_node.result_store.get_if_newer(shared_path_version)
        if new_path is not None:
            shared_path_version = new_version
            shared_path = list(new_path)
            shared_path_status = {} if status is None else dict(status)

            if shared_path_status.get("heuristic_ok", False) and shared_path_status.get("heuristic_refreshed", False):
                print(
                    f"Heuristic region ready. runs={shared_path_status['num_runs']}, "
                    f"points={shared_path_status['num_heuristic_points']}"
                )
            elif shared_path_status.get("heuristic_refreshed", False):
                if shared_path_status.get("fallback_corridor_used", False):
                    print(
                        "Heuristic prediction failed, fallback to corridor sampling. "
                        f"corridor_points={shared_path_status.get('fallback_corridor_points', 0)}"
                    )
                else:
                    print("Heuristic prediction failed, fallback to global sampling.")

            if not shared_path_status.get("plan_ok", False):
                print(
                    "Planning in progress (async anytime). "
                    f"tree_size={shared_path_status.get('tree_size', 0)}, "
                    f"budget={shared_path_status.get('iter_budget', 0)}"
                )
            else:
                path_debug_line_ids = update_path_debug_lines(
                    shared_path,
                    path_debug_line_ids,
                )
        
        # --- Execute Path ---
        # Hold one waypoint until reached, instead of switching every frame.
        if active_waypoint is None:
            while True:
                if len(shared_path) == 0:
                    break
                nxt = shared_path.pop(0)
                cand_waypoint, _ = nxt
                if np.linalg.norm(cand_waypoint - current_ee_pos) <= waypoint_reach_tol:
                    continue
                active_waypoint = cand_waypoint
                active_waypoint_last_progress_t = t
                active_waypoint_best_dist = float(np.linalg.norm(active_waypoint - current_ee_pos))
                ik_fail_streak = 0
                break

        if active_waypoint is not None:
            curr_wp_dist = float(np.linalg.norm(active_waypoint - current_ee_pos))
            if curr_wp_dist <= waypoint_reach_tol:
                active_waypoint = None
                ik_fail_streak = 0
                active_waypoint_best_dist = 1e9
            elif curr_wp_dist + 1e-4 < active_waypoint_best_dist:
                active_waypoint_best_dist = curr_wp_dist
                active_waypoint_last_progress_t = t
                ik_fail_streak = 0
            elif (t - active_waypoint_last_progress_t) > waypoint_stall_timeout:
                print(
                    "Waypoint stalled; dropping waypoint and forcing replan. "
                    f"dist={curr_wp_dist:.4f}, ik_fail_streak={ik_fail_streak}"
                )
                active_waypoint = None
                ik_fail_streak = 0
                active_waypoint_best_dist = 1e9
                shared_path = []
                last_replan_t = -1e9
            else:
                ee_to_wp = np.asarray(active_waypoint, dtype=np.float64) - current_ee_pos
                ee_dist = float(np.linalg.norm(ee_to_wp))
                if ee_dist > 1e-9:
                    ee_step = min(0.012, ee_dist)
                    ee_target = current_ee_pos + ee_to_wp / ee_dist * ee_step
                else:
                    ee_target = np.asarray(active_waypoint, dtype=np.float64)

                j_qs = solve_ik_continuous(
                    robotId,
                    end_effector_idx,
                    ee_target,
                    fixed_ee_orn,
                    joints_indices,
                )
                if j_qs is None:
                    ik_fail_streak += 1
                    hold_current_posture(robotId, joints_indices)
                else:
                    q_cmd = compute_limited_joint_targets(
                        robotId,
                        joints_indices,
                        j_qs,
                        max_step_rad=0.012,
                    )
                    if q_cmd is None:
                        ik_fail_streak += 1
                        hold_current_posture(robotId, joints_indices)
                    else:
                        ik_fail_streak = 0
                        p.setJointMotorControlArray(
                            robotId,
                            joints_indices,
                            p.POSITION_CONTROL,
                            targetPositions=q_cmd,
                            positionGains=[0.08] * 7,
                            velocityGains=[0.6] * 7,
                        )

                if ik_fail_streak >= max_ik_fail_streak:
                    print(
                        "IK failed repeatedly on active waypoint; dropping waypoint and forcing replan. "
                        f"ik_fail_streak={ik_fail_streak}"
                    )
                    active_waypoint = None
                    ik_fail_streak = 0
                    active_waypoint_best_dist = 1e9
                    shared_path = []
                    last_replan_t = -1e9
        else:
            hold_current_posture(robotId, joints_indices)

        dist_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        if dist_to_goal < 0.05:
            print("Goal Reached!")
            break
                
        if not safe_step_sim(client_id):
            print("Physics server disconnected during control loop; stopping.")
            break
        time.sleep(sim_dt)

    planning_node.stop()
    planning_node.join(timeout=1.0)
    pointcloud_node.stop()
    pointcloud_node.join(timeout=1.0)

    stats = planning_manager.get_stats_snapshot()
    print("=== Planning Stats Summary ===")
    print(f"replan_requests={stats['replan_requests']}")
    print(f"prediction_attempts={stats['prediction_attempts']}")
    print(f"heuristic_successes={stats['heuristic_successes']}")
    print(f"heuristic_failures={stats['heuristic_failures']}")
    print(f"heuristic_success_rate={stats['heuristic_success_rate']:.3f}")
    print(f"fallback_count={stats['fallback_count']}")
    print(f"plan_successes={stats['plan_successes']}")
    print(f"plan_failures={stats['plan_failures']}")
    print(f"avg_heuristic_points={stats['avg_heuristic_points']:.2f}")
    print(f"avg_png_runs={stats['avg_png_runs']:.2f}")
    print(f"total_waypoints_generated={stats['total_waypoints_generated']}")

    # Keep marker IDs referenced so they are not optimized away in refactoring.
    _ = (start_marker_id, goal_marker_id)

    if p.isConnected(client_id):
        p.disconnect(client_id)

if __name__ == "__main__":
    main()
