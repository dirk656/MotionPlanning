import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data

from pointcloud_tools import load_workspace_bounds


@dataclass
class RRTNode:
    pos: np.ndarray
    t: float
    parent: int


@dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float


class TimeBasedRRTPlanner:
    def __init__(
        self,
        bounds_min,
        bounds_max,
        speed,
        step_len,
        goal_tolerance,
        goal_bias,
        max_iters,
        time_horizon,
        collision_margin,
        dynamic_obstacle_fn,
    ):
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.speed = float(speed)
        self.step_len = float(step_len)
        self.goal_tolerance = float(goal_tolerance)
        self.goal_bias = float(goal_bias)
        self.max_iters = int(max_iters)
        self.time_horizon = float(time_horizon)
        self.collision_margin = float(collision_margin)
        self.dynamic_obstacle_fn = dynamic_obstacle_fn

    def _in_bounds(self, pos):
        return np.all(pos >= self.bounds_min) and np.all(pos <= self.bounds_max)

    def _is_state_valid(self, pos, t_abs):
        if not self._in_bounds(pos):
            return False
        for obs in self.dynamic_obstacle_fn(t_abs):
            if np.linalg.norm(pos - obs.center) <= (obs.radius + self.collision_margin):
                return False
        return True

    def _is_edge_valid(self, p0, t0, p1, t1):
        if t1 <= t0:
            return False
        seg_len = np.linalg.norm(p1 - p0)
        checks = max(3, int(math.ceil(seg_len / 0.02)))
        for alpha in np.linspace(0.0, 1.0, checks):
            pos = p0 + alpha * (p1 - p0)
            t_abs = t0 + alpha * (t1 - t0)
            if not self._is_state_valid(pos, t_abs):
                return False
        return True

    def _sample_pos(self, goal, rng):
        if rng.random() < self.goal_bias:
            return goal.copy()
        return rng.uniform(self.bounds_min, self.bounds_max)

    def _nearest_index(self, nodes, sample_pos):
        dists = [np.linalg.norm(n.pos - sample_pos) for n in nodes]
        return int(np.argmin(dists))

    def _extract_path(self, nodes, goal_idx):
        path = []
        idx = goal_idx
        while idx != -1:
            path.append((nodes[idx].pos.copy(), float(nodes[idx].t)))
            idx = nodes[idx].parent
        path.reverse()
        return path

    def plan(self, start_pos, goal_pos, start_time, rng):
        start_pos = np.asarray(start_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        if not self._is_state_valid(start_pos, start_time):
            return None

        nodes = [RRTNode(pos=start_pos, t=float(start_time), parent=-1)]

        for _ in range(self.max_iters):
            sample = self._sample_pos(goal_pos, rng)
            nearest_idx = self._nearest_index(nodes, sample)
            nearest = nodes[nearest_idx]

            direction = sample - nearest.pos
            dist = np.linalg.norm(direction)
            if dist < 1e-9:
                continue

            step = min(self.step_len, dist)
            new_pos = nearest.pos + direction / dist * step
            new_t = nearest.t + step / self.speed

            if new_t - start_time > self.time_horizon:
                continue
            if not self._is_edge_valid(nearest.pos, nearest.t, new_pos, new_t):
                continue

            nodes.append(RRTNode(pos=new_pos, t=new_t, parent=nearest_idx))
            new_idx = len(nodes) - 1

            to_goal = np.linalg.norm(goal_pos - new_pos)
            if to_goal <= self.goal_tolerance:
                goal_t = new_t + to_goal / self.speed
                if goal_t - start_time <= self.time_horizon and self._is_edge_valid(new_pos, new_t, goal_pos, goal_t):
                    nodes.append(RRTNode(pos=goal_pos.copy(), t=goal_t, parent=new_idx))
                    return self._extract_path(nodes, len(nodes) - 1)

            if to_goal <= 2.0 * self.step_len:
                goal_t = new_t + to_goal / self.speed
                if goal_t - start_time <= self.time_horizon and self._is_edge_valid(new_pos, new_t, goal_pos, goal_t):
                    nodes.append(RRTNode(pos=goal_pos.copy(), t=goal_t, parent=new_idx))
                    return self._extract_path(nodes, len(nodes) - 1)

        return None

    def remaining_path_valid(self, path, current_pos, current_t, next_idx):
        if path is None or next_idx >= len(path):
            return False
        prev_pos = np.asarray(current_pos, dtype=np.float64)
        prev_t = float(current_t)
        for i in range(next_idx, len(path)):
            nxt_pos, nxt_t = path[i]
            nxt_pos = np.asarray(nxt_pos, dtype=np.float64)
            nxt_t = max(float(nxt_t), prev_t + 1e-4)
            if not self._is_edge_valid(prev_pos, prev_t, nxt_pos, nxt_t):
                return False
            prev_pos = nxt_pos
            prev_t = nxt_t
        return True


def project_to_sphere(candidate_pos, center, radius):
    offset = candidate_pos - center
    dist = np.linalg.norm(offset)
    if dist <= radius:
        return candidate_pos
    return center + offset / dist * radius


def dynamic_obstacle_center(t_abs):
    raw = np.array([0.6 + 0.3 * np.sin(t_abs), 0.0, 0.5], dtype=np.float64)
    return project_to_sphere(raw, np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.5)


def make_dynamic_obstacles(t_abs, obstacle_radius):
    center = dynamic_obstacle_center(t_abs)
    return [SphereObstacle(center=center, radius=obstacle_radius)]


def draw_path_debug(path):
    debug_ids = []
    if path is None or len(path) < 2:
        return debug_ids
    for i in range(len(path) - 1):
        p0 = path[i][0]
        p1 = path[i + 1][0]
        debug_ids.append(
            p.addUserDebugLine(
                p0,
                p1,
                lineColorRGB=[0.1, 0.8, 1.0],
                lineWidth=2.0,
                lifeTime=0,
            )
        )
    return debug_ids


def create_marker(position, radius, rgba):
    vid = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    cid = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition=position,
    )


def main():
    parser = argparse.ArgumentParser(description="Time-based RRT with online replanning under dynamic obstacle motion.")
    parser.add_argument("--gui", dest="gui", action="store_true", help="Run with GUI (default).")
    parser.add_argument("--headless", dest="gui", action="store_false", help="Run without GUI (DIRECT).")
    parser.set_defaults(gui=True)
    parser.add_argument("--env_config", type=str, default="src/config/env.yaml", help="YAML file containing workspace_bounds.")
    parser.add_argument("--start", nargs=3, type=float, default=[-0.35, -0.25, 0.2], help="Start position x y z.")
    parser.add_argument("--goal", nargs=3, type=float, default=[0.35, 0.25, 0.35], help="Goal position x y z.")
    parser.add_argument("--sim_time", type=float, default=15.0, help="Max simulation time in seconds.")
    parser.add_argument("--sim_dt", type=float, default=1.0 / 30.0, help="Simulation/control time step.")
    parser.add_argument("--speed", type=float, default=0.25, help="Agent speed in m/s.")
    parser.add_argument("--step_len", type=float, default=0.08, help="RRT extension step length in meters.")
    parser.add_argument("--goal_tolerance", type=float, default=0.08, help="Goal position tolerance in meters.")
    parser.add_argument("--goal_bias", type=float, default=0.25, help="Goal sampling probability for RRT.")
    parser.add_argument("--max_iters", type=int, default=2500, help="Max iterations for each replan cycle.")
    parser.add_argument("--time_horizon", type=float, default=10.0, help="Planning horizon in seconds.")
    parser.add_argument("--collision_margin", type=float, default=0.05, help="Safety margin around dynamic obstacle.")
    parser.add_argument("--replan_interval", type=float, default=0.5, help="Periodic online replanning interval in seconds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for planner sampling.")
    args = parser.parse_args()

    bounds_min, bounds_max = load_workspace_bounds(args.env_config)
    start = np.asarray(args.start, dtype=np.float64)
    goal = np.asarray(args.goal, dtype=np.float64)

    if args.gui:
        client_id = p.connect(p.GUI)
        use_gui = client_id >= 0
        if not use_gui:
            print("Warning: GUI unavailable, fallback to DIRECT mode")
            client_id = p.connect(p.DIRECT)
    else:
        client_id = p.connect(p.DIRECT)
        use_gui = False

    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    if use_gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.35,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.0, 0.0, 0.35],
        )

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    box_size = 0.2
    dyn_obs_radius = math.sqrt(3.0) * (box_size * 0.5)
    obs_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[box_size * 0.5] * 3,
        rgbaColor=[1.0, 0.2, 0.2, 1.0],
    )
    obs_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_size * 0.5] * 3)
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obs_collision,
        baseVisualShapeIndex=obs_visual,
        basePosition=dynamic_obstacle_center(0.0).tolist(),
    )

    start_id = create_marker(start.tolist(), radius=0.03, rgba=[0.2, 0.9, 0.2, 1.0])
    goal_id = create_marker(goal.tolist(), radius=0.03, rgba=[1.0, 0.9, 0.1, 1.0])
    agent_id = create_marker(start.tolist(), radius=0.025, rgba=[0.1, 0.7, 1.0, 1.0])

    planner = TimeBasedRRTPlanner(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        speed=args.speed,
        step_len=args.step_len,
        goal_tolerance=args.goal_tolerance,
        goal_bias=args.goal_bias,
        max_iters=args.max_iters,
        time_horizon=args.time_horizon,
        collision_margin=args.collision_margin,
        dynamic_obstacle_fn=lambda t_abs: make_dynamic_obstacles(t_abs, dyn_obs_radius),
    )

    rng = np.random.default_rng(args.seed)
    sim_t = 0.0
    agent_pos = start.copy()
    last_plan_t = -1e9
    active_path = None
    next_idx = 1
    debug_line_ids = []

    reached = False
    try:
        max_steps = int(args.sim_time / args.sim_dt)
        for _ in range(max_steps):
            obs_center = dynamic_obstacle_center(sim_t)
            p.resetBasePositionAndOrientation(obstacle_id, obs_center.tolist(), [0, 0, 0, 1])

            periodic_replan = (sim_t - last_plan_t) >= args.replan_interval
            path_invalid = not planner.remaining_path_valid(active_path, agent_pos, sim_t, next_idx)
            need_replan = active_path is None or periodic_replan or path_invalid

            if need_replan:
                new_path = planner.plan(agent_pos, goal, sim_t, rng)
                if new_path is not None:
                    active_path = new_path
                    next_idx = 1
                    last_plan_t = sim_t

                    for line_id in debug_line_ids:
                        p.removeUserDebugItem(line_id)
                    debug_line_ids = draw_path_debug(active_path)

            if active_path is not None and next_idx < len(active_path):
                target_pos, _ = active_path[next_idx]
                to_target = target_pos - agent_pos
                dist = np.linalg.norm(to_target)
                max_step = args.speed * args.sim_dt
                if dist <= max_step:
                    agent_pos = target_pos.copy()
                    next_idx += 1
                else:
                    agent_pos = agent_pos + to_target / dist * max_step

            p.resetBasePositionAndOrientation(agent_id, agent_pos.tolist(), [0, 0, 0, 1])

            if np.linalg.norm(agent_pos - goal) <= args.goal_tolerance:
                reached = True
                break

            p.stepSimulation()
            if use_gui:
                time.sleep(args.sim_dt)
            sim_t += args.sim_dt

    finally:
        if reached:
            print(f"Success: goal reached at t={sim_t:.2f}s")
        else:
            print(f"Finished without reaching goal, last distance={np.linalg.norm(agent_pos - goal):.3f} m")
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
