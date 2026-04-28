import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p

from src.planning_utils.collision_check_utils import points_in_AABB_3d


@dataclass
class TimeRRTNode:
    pos: np.ndarray
    t_abs: float
    parent: int


@dataclass
class SphereObstacle:
    center: np.ndarray
    radius: float


@dataclass
class SubTreeState:
    nodes: list
    root_pos: np.ndarray
    root_t: float


class HeuristicTimeRRTPlanner:
    """Time-based RRT with heuristic-region biased sampling."""

    def __init__(
        self,
        bounds_min,
        bounds_max,
        speed=0.5,
        step_len=0.08,
        goal_tolerance=0.05,
        goal_bias=0.1,
        heuristic_bias=0.75,
        max_iters=2500,
        time_horizon=6.0,
        collision_margin=0.03,
        max_subtrees=4,
        subtree_spawn_prob=0.12,
        subtree_merge_distance=0.12,
        risk_lookahead_s=0.35,
        risk_samples=3,
        risk_clearance_scale=1.0,
        singularity_threshold=None,
        singularity_penalty_gain=2.0,
        failed_region_radius=0.10,
        failed_region_spawn_threshold=3,
    ):
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.speed = float(speed)
        self.step_len = float(step_len)
        self.goal_tolerance = float(goal_tolerance)
        self.goal_bias = float(goal_bias)
        self.heuristic_bias = float(heuristic_bias)
        self.max_iters = int(max_iters)
        self.time_horizon = float(time_horizon)
        self.collision_margin = float(collision_margin)
        self.base_heuristic_bias = float(heuristic_bias)
        self.runtime_heuristic_bias = float(heuristic_bias)
        self.max_subtrees = int(max(0, max_subtrees))
        self.subtree_spawn_prob = float(np.clip(subtree_spawn_prob, 0.0, 1.0))
        self.subtree_merge_distance = float(max(subtree_merge_distance, 0.0))
        self.risk_lookahead_s = float(max(risk_lookahead_s, 0.0))
        self.risk_samples = int(max(1, risk_samples))
        self.risk_clearance_scale = float(max(risk_clearance_scale, 0.0))
        self.singularity_threshold = None if singularity_threshold is None else float(singularity_threshold)
        self.singularity_penalty_gain = float(max(singularity_penalty_gain, 0.0))
        self.failed_region_radius = float(max(failed_region_radius, 1e-4))
        self.failed_region_spawn_threshold = int(max(1, failed_region_spawn_threshold))
        self.heuristic_points = None
        self.static_obs_aabb = None

    def set_runtime_heuristic_bias(self, heuristic_bias):
        self.runtime_heuristic_bias = float(np.clip(float(heuristic_bias), 0.0, 1.0))

    def set_heuristic_points(self, heuristic_points):
        if heuristic_points is None:
            self.heuristic_points = None
            return

        heuristic_points = np.asarray(heuristic_points, dtype=np.float64)
        if heuristic_points.ndim != 2 or heuristic_points.shape[1] != 3:
            raise ValueError("heuristic_points must have shape [M, 3]")
        if heuristic_points.shape[0] == 0:
            self.heuristic_points = None
            return

        self.heuristic_points = heuristic_points

    def set_static_aabb_obstacles(self, obs_aabb):
        if obs_aabb is None:
            self.static_obs_aabb = None
            return

        obs_aabb = np.asarray(obs_aabb, dtype=np.float64)
        if obs_aabb.ndim != 2 or obs_aabb.shape[1] != 6:
            raise ValueError("obs_aabb must have shape [K, 6]")
        if obs_aabb.shape[0] == 0:
            self.static_obs_aabb = None
            return
        self.static_obs_aabb = obs_aabb

    def _in_bounds(self, pos):
        return np.all(pos >= self.bounds_min) and np.all(pos <= self.bounds_max)

    def _risk_collision_likely(self, pos, t_abs, dynamic_obstacle_fn=None):
        if dynamic_obstacle_fn is None or self.risk_lookahead_s <= 0.0:
            return False

        lookahead = np.linspace(0.0, self.risk_lookahead_s, self.risk_samples)
        for dt in lookahead:
            obstacles = dynamic_obstacle_fn(float(t_abs + dt))
            for obs in obstacles:
                clearance = float(obs.radius) + self.collision_margin * self.risk_clearance_scale
                if np.linalg.norm(pos - obs.center) <= clearance:
                    return True
        return False

    def _is_state_valid(self, pos, t_abs, dynamic_obstacle_fn=None, arm_collision_fn=None, arm_singularity_fn=None):
        if not self._in_bounds(pos):
            return False

        if self.static_obs_aabb is not None:
            if points_in_AABB_3d(tuple(pos.tolist()), self.static_obs_aabb, clearance=self.collision_margin):
                return False

        if arm_collision_fn is not None and arm_collision_fn(pos, t_abs):
            return False

        if arm_singularity_fn is not None and self.singularity_threshold is not None:
            cond_num = float(arm_singularity_fn(pos, t_abs))
            if not np.isfinite(cond_num):
                return False

        if self._risk_collision_likely(pos, t_abs, dynamic_obstacle_fn=dynamic_obstacle_fn):
            return False

        if dynamic_obstacle_fn is None:
            return True

        obstacles = dynamic_obstacle_fn(t_abs)
        for obs in obstacles:
            if np.linalg.norm(pos - obs.center) <= obs.radius + self.collision_margin:
                return False
        return True

    def _is_edge_valid(self, p0, t0, p1, t1, dynamic_obstacle_fn=None, arm_collision_fn=None, arm_singularity_fn=None):
        if t1 <= t0:
            return False

        seg_len = np.linalg.norm(p1 - p0)
        checks = max(3, int(math.ceil(seg_len / 0.02)))

        for alpha in np.linspace(0.0, 1.0, checks):
            pos = p0 + alpha * (p1 - p0)
            t_abs = t0 + alpha * (t1 - t0)
            if not self._is_state_valid(
                pos,
                t_abs,
                dynamic_obstacle_fn=dynamic_obstacle_fn,
                arm_collision_fn=arm_collision_fn,
                arm_singularity_fn=arm_singularity_fn,
            ):
                return False
        return True

    def _sample(self, goal, rng):
        r = rng.random()
        if r < self.goal_bias:
            return goal.copy()

        if self.heuristic_points is not None and r < self.goal_bias + self.runtime_heuristic_bias:
            idx = rng.integers(0, self.heuristic_points.shape[0])
            base = self.heuristic_points[idx]
            noise = rng.normal(0.0, self.step_len * 0.15, size=3)
            sample = base + noise
            return np.clip(sample, self.bounds_min, self.bounds_max)

        return rng.uniform(self.bounds_min, self.bounds_max)

    @staticmethod
    def _nearest_index(nodes, sample):
        dists = [np.linalg.norm(n.pos - sample) for n in nodes]
        return int(np.argmin(dists))

    @staticmethod
    def _extract_path(nodes, goal_idx):
        path = []
        idx = goal_idx
        while idx != -1:
            node = nodes[idx]
            path.append((node.pos.copy(), float(node.t_abs)))
            idx = node.parent
        path.reverse()
        return path

    @staticmethod
    def init_tree(start, start_time):
        start = np.asarray(start, dtype=np.float64)
        start_time = float(start_time)
        return [TimeRRTNode(pos=start, t_abs=start_time, parent=-1)]

    @staticmethod
    def _sample_subtree_guidance(subtree_nodes, anchor_idx, rng):
        node = subtree_nodes[anchor_idx]
        if node.parent != -1 and rng.random() < 0.6:
            parent = subtree_nodes[node.parent]
            direction = parent.pos - node.pos
            dist = np.linalg.norm(direction)
            if dist > 1e-9:
                return node.pos + direction / dist * min(0.08, dist)
        return node.pos.copy()

    def _singularity_penalty_scale(self, pos, t_abs, arm_singularity_fn=None):
        if arm_singularity_fn is None or self.singularity_threshold is None:
            return 1.0
        cond_num = float(arm_singularity_fn(pos, t_abs))
        if not np.isfinite(cond_num):
            return 0.0
        if cond_num <= self.singularity_threshold:
            return 1.0
        ratio = cond_num / max(self.singularity_threshold, 1e-6)
        # Soft penalty: shrink step near singular region instead of hard reject.
        scale = 1.0 / (1.0 + self.singularity_penalty_gain * (ratio - 1.0))
        return float(np.clip(scale, 0.15, 1.0))

    def _register_failed_region(self, failed_regions, sample_pos, t_abs):
        sample_pos = np.asarray(sample_pos, dtype=np.float64)
        for region in failed_regions:
            if np.linalg.norm(region["pos"] - sample_pos) <= self.failed_region_radius:
                region["count"] += 1
                region["t_abs"] = float(t_abs)
                region["pos"] = 0.7 * region["pos"] + 0.3 * sample_pos
                return
        failed_regions.append({"pos": sample_pos.copy(), "count": 1, "t_abs": float(t_abs)})

    def _pop_forced_spawn_candidate(self, failed_regions, dynamic_obstacle_fn=None):
        if len(failed_regions) == 0:
            return None
        best_idx = -1
        best_score = -1.0
        for i, region in enumerate(failed_regions):
            pos = region["pos"]
            t_abs = float(region["t_abs"])
            risk_bonus = 1.0 if self._risk_collision_likely(pos, t_abs, dynamic_obstacle_fn=dynamic_obstacle_fn) else 0.0
            score = float(region["count"]) + 2.0 * risk_bonus
            if score > best_score:
                best_idx = i
                best_score = score
        if best_idx < 0:
            return None
        candidate = failed_regions[best_idx]
        if int(candidate["count"]) < self.failed_region_spawn_threshold:
            return None
        failed_regions.pop(best_idx)
        return np.asarray(candidate["pos"], dtype=np.float64)

    @staticmethod
    def _path_to_root(nodes, idx):
        path = []
        cur = int(idx)
        while cur != -1:
            path.append(cur)
            cur = int(nodes[cur].parent)
        return path

    def _try_bidirectional_bridge(
        self,
        root_tree_nodes,
        subtree,
        goal,
        dynamic_obstacle_fn,
        arm_collision_fn,
        arm_singularity_fn,
    ):
        if len(root_tree_nodes) == 0 or len(subtree.nodes) == 0:
            return None

        # Find nearest pair between root tree and subtree.
        best = None
        for i, rn in enumerate(root_tree_nodes):
            for j, sn in enumerate(subtree.nodes):
                d = float(np.linalg.norm(rn.pos - sn.pos))
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None or best[0] > self.subtree_merge_distance:
            return None

        _, root_idx, sub_idx = best
        root_node = root_tree_nodes[root_idx]
        sub_node = subtree.nodes[sub_idx]
        bridge_len = float(np.linalg.norm(sub_node.pos - root_node.pos))
        bridge_t = float(root_node.t_abs + bridge_len / max(self.speed, 1e-6))
        if (bridge_t - float(root_tree_nodes[0].t_abs)) > self.time_horizon:
            return None
        if not self._is_edge_valid(
            root_node.pos,
            root_node.t_abs,
            sub_node.pos,
            bridge_t,
            dynamic_obstacle_fn=dynamic_obstacle_fn,
            arm_collision_fn=arm_collision_fn,
            arm_singularity_fn=arm_singularity_fn,
        ):
            return None

        # Graft a subtree branch to root tree: sub anchor -> ... -> subtree root.
        new_parent = root_idx
        last_pos = root_node.pos
        last_t = float(root_node.t_abs)
        chain = self._path_to_root(subtree.nodes, sub_idx)
        for idx in chain:
            pos = np.asarray(subtree.nodes[idx].pos, dtype=np.float64)
            dt = float(np.linalg.norm(pos - last_pos)) / max(self.speed, 1e-6)
            next_t = float(last_t + dt)
            if (next_t - float(root_tree_nodes[0].t_abs)) > self.time_horizon:
                break
            if not self._is_edge_valid(
                last_pos,
                last_t,
                pos,
                next_t,
                dynamic_obstacle_fn=dynamic_obstacle_fn,
                arm_collision_fn=arm_collision_fn,
                arm_singularity_fn=arm_singularity_fn,
            ):
                break
            root_tree_nodes.append(TimeRRTNode(pos=pos.copy(), t_abs=next_t, parent=new_parent))
            new_parent = len(root_tree_nodes) - 1
            last_pos = pos
            last_t = next_t

        # If graft tail can directly connect to goal, return solved path.
        dist_goal = float(np.linalg.norm(goal - last_pos))
        if dist_goal <= self.goal_tolerance:
            goal_t = float(last_t + dist_goal / max(self.speed, 1e-6))
            if self._is_edge_valid(
                last_pos,
                last_t,
                goal,
                goal_t,
                dynamic_obstacle_fn=dynamic_obstacle_fn,
                arm_collision_fn=arm_collision_fn,
                arm_singularity_fn=arm_singularity_fn,
            ):
                root_tree_nodes.append(TimeRRTNode(pos=goal.copy(), t_abs=goal_t, parent=new_parent))
                return self._extract_path(root_tree_nodes, len(root_tree_nodes) - 1)

        return None

    def _expand_once(
        self,
        tree_nodes,
        sample,
        root_t,
        dynamic_obstacle_fn,
        arm_collision_fn,
        arm_singularity_fn,
    ):
        nearest_idx = self._nearest_index(tree_nodes, sample)
        nearest = tree_nodes[nearest_idx]

        direction = sample - nearest.pos
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            return None, None

        step = min(self.step_len, dist)
        new_pos = nearest.pos + direction / dist * step
        new_t = nearest.t_abs + step / self.speed

        step_scale = self._singularity_penalty_scale(new_pos, new_t, arm_singularity_fn=arm_singularity_fn)
        if step_scale <= 0.0:
            return None, None
        if step_scale < 0.999:
            step = max(1e-4, step * step_scale)
            new_pos = nearest.pos + direction / dist * step
            new_t = nearest.t_abs + step / self.speed

        if (new_t - root_t) > self.time_horizon:
            return None, None

        if not self._is_edge_valid(
            nearest.pos,
            nearest.t_abs,
            new_pos,
            new_t,
            dynamic_obstacle_fn=dynamic_obstacle_fn,
            arm_collision_fn=arm_collision_fn,
            arm_singularity_fn=arm_singularity_fn,
        ):
            return None, None

        tree_nodes.append(TimeRRTNode(pos=new_pos, t_abs=new_t, parent=nearest_idx))
        new_idx = len(tree_nodes) - 1
        return new_idx, new_pos

    def _spawn_subtree(
        self,
        subtrees,
        start_time,
        goal,
        rng,
        dynamic_obstacle_fn,
        arm_collision_fn,
        arm_singularity_fn,
        forced_seed_pos=None,
    ):
        if self.max_subtrees <= 0 or len(subtrees) >= self.max_subtrees:
            return

        if forced_seed_pos is None:
            sample = self._sample(goal, rng)
        else:
            sample = np.asarray(forced_seed_pos, dtype=np.float64)
            sample = np.clip(sample, self.bounds_min, self.bounds_max)
        root_t = float(start_time + rng.uniform(0.0, max(self.time_horizon * 0.35, 1e-3)))
        if not self._is_state_valid(
            sample,
            root_t,
            dynamic_obstacle_fn=dynamic_obstacle_fn,
            arm_collision_fn=arm_collision_fn,
            arm_singularity_fn=arm_singularity_fn,
        ):
            return
        nodes = self.init_tree(sample, root_t)
        subtrees.append(SubTreeState(nodes=nodes, root_pos=sample.copy(), root_t=root_t))

    def plan_incremental(
        self,
        start,
        goal,
        start_time,
        tree_nodes=None,
        subtrees=None,
        iter_budget=80,
        dynamic_obstacle_fn=None,
        arm_collision_fn=None,
        arm_singularity_fn=None,
        rng=None,
    ):
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        start_time = float(start_time)
        iter_budget = int(max(1, iter_budget))

        if rng is None:
            rng = np.random.default_rng()

        if tree_nodes is None or len(tree_nodes) == 0:
            if not self._is_state_valid(
                start,
                start_time,
                dynamic_obstacle_fn=dynamic_obstacle_fn,
                arm_collision_fn=arm_collision_fn,
                arm_singularity_fn=arm_singularity_fn,
            ):
                return None, subtrees, None
            tree_nodes = self.init_tree(start, start_time)

        if subtrees is None:
            subtrees = []
        failed_regions = []

        root = tree_nodes[0]
        if (float(start_time) + 1e-6) < float(root.t_abs):
            return None, subtrees, None

        for _ in range(iter_budget):
            forced_seed = self._pop_forced_spawn_candidate(
                failed_regions,
                dynamic_obstacle_fn=dynamic_obstacle_fn,
            )
            if rng.random() < self.subtree_spawn_prob:
                self._spawn_subtree(
                    subtrees=subtrees,
                    start_time=start_time,
                    goal=goal,
                    rng=rng,
                    dynamic_obstacle_fn=dynamic_obstacle_fn,
                    arm_collision_fn=arm_collision_fn,
                    arm_singularity_fn=arm_singularity_fn,
                    forced_seed_pos=forced_seed,
                )
            elif forced_seed is not None:
                self._spawn_subtree(
                    subtrees=subtrees,
                    start_time=start_time,
                    goal=goal,
                    rng=rng,
                    dynamic_obstacle_fn=dynamic_obstacle_fn,
                    arm_collision_fn=arm_collision_fn,
                    arm_singularity_fn=arm_singularity_fn,
                    forced_seed_pos=forced_seed,
                )

            for subtree in subtrees:
                sub_sample = rng.uniform(self.bounds_min, self.bounds_max)
                self._expand_once(
                    tree_nodes=subtree.nodes,
                    sample=sub_sample,
                    root_t=subtree.root_t,
                    dynamic_obstacle_fn=dynamic_obstacle_fn,
                    arm_collision_fn=arm_collision_fn,
                    arm_singularity_fn=arm_singularity_fn,
                )

            sample = self._sample(goal, rng)
            new_idx, new_pos = self._expand_once(
                tree_nodes=tree_nodes,
                sample=sample,
                root_t=root.t_abs,
                dynamic_obstacle_fn=dynamic_obstacle_fn,
                arm_collision_fn=arm_collision_fn,
                arm_singularity_fn=arm_singularity_fn,
            )
            if new_idx is None:
                self._register_failed_region(failed_regions, sample_pos=sample, t_abs=start_time)
                continue

            new_t = tree_nodes[new_idx].t_abs

            # Subtree-guided local refinement when root tree gets close to any subtree node.
            for subtree in subtrees:
                if len(subtree.nodes) == 0:
                    continue
                dists = [np.linalg.norm(n.pos - new_pos) for n in subtree.nodes]
                anchor_idx = int(np.argmin(dists))
                if float(dists[anchor_idx]) > self.subtree_merge_distance:
                    continue
                guide_sample = self._sample_subtree_guidance(subtree.nodes, anchor_idx, rng)
                self._expand_once(
                    tree_nodes=tree_nodes,
                    sample=guide_sample,
                    root_t=root.t_abs,
                    dynamic_obstacle_fn=dynamic_obstacle_fn,
                    arm_collision_fn=arm_collision_fn,
                    arm_singularity_fn=arm_singularity_fn,
                )

                bridged_path = self._try_bidirectional_bridge(
                    root_tree_nodes=tree_nodes,
                    subtree=subtree,
                    goal=goal,
                    dynamic_obstacle_fn=dynamic_obstacle_fn,
                    arm_collision_fn=arm_collision_fn,
                    arm_singularity_fn=arm_singularity_fn,
                )
                if bridged_path is not None:
                    return tree_nodes, subtrees, bridged_path

            to_goal = np.linalg.norm(goal - new_pos)
            if to_goal <= self.goal_tolerance:
                goal_t = new_t + to_goal / self.speed
                if (
                    (goal_t - root.t_abs) <= self.time_horizon
                    and self._is_edge_valid(
                        new_pos,
                        new_t,
                        goal,
                        goal_t,
                        dynamic_obstacle_fn=dynamic_obstacle_fn,
                        arm_collision_fn=arm_collision_fn,
                        arm_singularity_fn=arm_singularity_fn,
                    )
                ):
                    tree_nodes.append(TimeRRTNode(pos=goal.copy(), t_abs=goal_t, parent=new_idx))
                    return tree_nodes, subtrees, self._extract_path(tree_nodes, len(tree_nodes) - 1)

        return tree_nodes, subtrees, None

    def plan(
        self,
        start,
        goal,
        start_time,
        dynamic_obstacle_fn=None,
        arm_collision_fn=None,
        rng=None,
    ):
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        start_time = float(start_time)

        nodes, _, path = self.plan_incremental(
            start=start,
            goal=goal,
            start_time=start_time,
            tree_nodes=None,
            subtrees=None,
            iter_budget=self.max_iters,
            dynamic_obstacle_fn=dynamic_obstacle_fn,
            arm_collision_fn=arm_collision_fn,
            arm_singularity_fn=None,
            rng=rng,
        )
        _ = nodes
        return path

    def remaining_path_valid(
        self,
        path,
        current_pos,
        current_t,
        dynamic_obstacle_fn=None,
        arm_collision_fn=None,
    ):
        if path is None or len(path) < 2:
            return False

        prev_pos = np.asarray(current_pos, dtype=np.float64)
        prev_t = float(current_t)
        for nxt_pos, nxt_t in path:
            nxt_pos = np.asarray(nxt_pos, dtype=np.float64)
            nxt_t = max(float(nxt_t), prev_t + 1e-4)
            if not self._is_edge_valid(
                prev_pos,
                prev_t,
                nxt_pos,
                nxt_t,
                dynamic_obstacle_fn=dynamic_obstacle_fn,
                arm_collision_fn=arm_collision_fn,
            ):
                return False
            prev_pos = nxt_pos
            prev_t = nxt_t

        return True


class PyBulletArmCollisionChecker:
    """Check whole-arm collision by solving IK then querying PyBullet distances."""

    def __init__(
        self,
        robot_id,
        end_effector_idx,
        controlled_joint_indices,
        obstacle_body_ids,
        safety_distance=0.01,
    ):
        self.robot_id = int(robot_id)
        self.end_effector_idx = int(end_effector_idx)
        self.controlled_joint_indices = list(controlled_joint_indices)
        self.obstacle_body_ids = list(obstacle_body_ids)
        self.safety_distance = float(safety_distance)

    def is_collision_at_ee(self, ee_pos):
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        if not p.isConnected():
            return False

        try:
            old_states = [p.getJointState(self.robot_id, j)[0] for j in self.controlled_joint_indices]
            ik_solution = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_idx,
                ee_pos.tolist(),
            )

            if len(ik_solution) < len(self.controlled_joint_indices):
                return False

            for local_i, joint_idx in enumerate(self.controlled_joint_indices):
                p.resetJointState(self.robot_id, joint_idx, float(ik_solution[local_i]))

            p.performCollisionDetection()

            in_collision = False
            for obs_id in self.obstacle_body_ids:
                closest = p.getClosestPoints(
                    bodyA=self.robot_id,
                    bodyB=obs_id,
                    distance=self.safety_distance,
                )
                if len(closest) > 0:
                    in_collision = True
                    break

            for local_i, joint_idx in enumerate(self.controlled_joint_indices):
                p.resetJointState(self.robot_id, joint_idx, float(old_states[local_i]))

            return in_collision
        except Exception:
            # If simulation state is transiently unavailable, skip this check to keep planner alive.
            return False


class OnlineHeuristicPlanningManager:
    """Owns online replanning state; main loop only feeds observations and executes waypoints."""

    def __init__(
        self,
        planner,
        predictor,
        dynamic_obstacle_fn,
        arm_collision_checker=None,
        neighbor_radius=0.1,
        max_trial_attempts=3,
        incremental_iter_budget=80,
        tree_resume_pos_tol=0.08,
        tree_resume_time_window=0.35,
        heuristic_update_interval=0.15,
        bias_decay=0.85,
        bias_recover=0.05,
        bias_decay_fail_threshold=3,
        min_heuristic_bias=0.05,
        path_smoothing_trials=20,
        path_smoothing_enabled=True,
        arm_singularity_checker=None,
        stuck_time_s=10.0,
        stuck_goal_improve_eps=0.01,
        rng=None,
    ):
        self.planner = planner
        self.predictor = predictor
        self.dynamic_obstacle_fn = dynamic_obstacle_fn
        self.arm_collision_checker = arm_collision_checker
        self.neighbor_radius = float(neighbor_radius)
        self.max_trial_attempts = int(max_trial_attempts)
        self.incremental_iter_budget = int(max(1, incremental_iter_budget))
        self.tree_resume_pos_tol = float(tree_resume_pos_tol)
        self.tree_resume_time_window = float(tree_resume_time_window)
        self.heuristic_update_interval = float(heuristic_update_interval)
        self.bias_decay = float(np.clip(bias_decay, 0.1, 0.99))
        self.bias_recover = float(max(bias_recover, 0.0))
        self.bias_decay_fail_threshold = int(max(1, bias_decay_fail_threshold))
        self.min_heuristic_bias = float(np.clip(min_heuristic_bias, 0.0, 1.0))
        self.path_smoothing_trials = int(max(0, path_smoothing_trials))
        self.path_smoothing_enabled = bool(path_smoothing_enabled)
        self.arm_singularity_checker = arm_singularity_checker
        self.stuck_time_s = float(max(stuck_time_s, 0.1))
        self.stuck_goal_improve_eps = float(max(stuck_goal_improve_eps, 1e-6))
        self.rng = np.random.default_rng() if rng is None else rng

        self.current_path = []
        self._active_tree_nodes = None
        self._active_tree_start_pos = None
        self._active_tree_start_t = None
        self._active_tree_goal_pos = None
        self._active_subtrees = []
        self._last_heuristic_update_t = -1e9
        self._consecutive_plan_failures = 0
        self._best_goal_dist = float("inf")
        self._last_progress_t = -1e9
        self._adaptive_heuristic_bias = float(self.planner.base_heuristic_bias)
        self.planner.set_runtime_heuristic_bias(self._adaptive_heuristic_bias)
        self.stats = {
            "replan_requests": 0,
            "prediction_attempts": 0,
            "heuristic_successes": 0,
            "heuristic_failures": 0,
            "fallback_count": 0,
            "plan_successes": 0,
            "plan_failures": 0,
            "total_num_runs": 0,
            "total_heuristic_points": 0,
            "total_waypoints_generated": 0,
            "subtree_expansions": 0,
            "bias_decay_events": 0,
            "smoothing_applied": 0,
        }

    def _arm_collision_fn(self, pos, t_abs):
        if self.arm_collision_checker is None:
            return False
        return self.arm_collision_checker.is_collision_at_ee(pos)

    def _arm_singularity_fn(self, pos, t_abs):
        _ = t_abs
        if self.arm_singularity_checker is None:
            return 0.0
        try:
            return float(self.arm_singularity_checker(pos))
        except Exception:
            return float("inf")

    def _maybe_decay_bias(self, force=False):
        if (not force) and self._consecutive_plan_failures < self.bias_decay_fail_threshold:
            return
        prev = self._adaptive_heuristic_bias
        self._adaptive_heuristic_bias = max(self.min_heuristic_bias, self._adaptive_heuristic_bias * self.bias_decay)
        if self._adaptive_heuristic_bias < prev - 1e-9:
            self.stats["bias_decay_events"] += 1
        self.planner.set_runtime_heuristic_bias(self._adaptive_heuristic_bias)

    def _recover_bias_on_success(self):
        self._adaptive_heuristic_bias = min(
            self.planner.base_heuristic_bias,
            self._adaptive_heuristic_bias + self.bias_recover,
        )
        self.planner.set_runtime_heuristic_bias(self._adaptive_heuristic_bias)

    def _update_stuck_state(self, current_pos, goal_pos, current_t):
        goal_dist = float(np.linalg.norm(np.asarray(goal_pos, dtype=np.float64) - np.asarray(current_pos, dtype=np.float64)))
        if goal_dist + self.stuck_goal_improve_eps < self._best_goal_dist:
            self._best_goal_dist = goal_dist
            self._last_progress_t = float(current_t)
            return

        if self._last_progress_t < -1e8:
            self._last_progress_t = float(current_t)
            self._best_goal_dist = goal_dist
            return

        if (float(current_t) - float(self._last_progress_t)) >= self.stuck_time_s:
            self._maybe_decay_bias(force=True)
            self._last_progress_t = float(current_t)

    def _shortcut_smooth_path(self, path, current_t):
        if (not self.path_smoothing_enabled) or self.path_smoothing_trials <= 0 or path is None or len(path) < 3:
            return path

        smoothed = list(path)
        for _ in range(self.path_smoothing_trials):
            if len(smoothed) < 3:
                break
            i = int(self.rng.integers(0, len(smoothed) - 2))
            j = int(self.rng.integers(i + 2, len(smoothed)))
            p_i = np.asarray(smoothed[i][0], dtype=np.float64)
            p_j = np.asarray(smoothed[j][0], dtype=np.float64)
            t_i = float(smoothed[i][1])
            seg_len = float(np.linalg.norm(p_j - p_i))
            t_j = max(float(smoothed[j][1]), t_i + seg_len / max(self.planner.speed, 1e-6))

            if not self.planner._is_edge_valid(
                p_i,
                t_i,
                p_j,
                t_j,
                dynamic_obstacle_fn=self.dynamic_obstacle_fn,
                arm_collision_fn=self._arm_collision_fn,
                arm_singularity_fn=self._arm_singularity_fn,
            ):
                continue

            smoothed = smoothed[: i + 1] + smoothed[j:]

        # Re-time with constant speed so controller keeps monotonic timestamps.
        base_t = max(float(current_t), float(smoothed[0][1]))
        out = []
        prev_p = np.asarray(smoothed[0][0], dtype=np.float64)
        out.append((prev_p.copy(), base_t))
        t_acc = base_t
        for k in range(1, len(smoothed)):
            p_k = np.asarray(smoothed[k][0], dtype=np.float64)
            t_acc += float(np.linalg.norm(p_k - prev_p)) / max(self.planner.speed, 1e-6)
            out.append((p_k.copy(), t_acc))
            prev_p = p_k
        return out

    def _build_corridor_heuristic_points(self, start_pos, goal_pos, n_points=96, radius=0.04):
        start_pos = np.asarray(start_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        seg = goal_pos - start_pos
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-8:
            return np.empty((0, 3), dtype=np.float64)

        seg_dir = seg / seg_len
        basis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(seg_dir, basis))) > 0.95:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        n1 = np.cross(seg_dir, basis)
        n1_norm = np.linalg.norm(n1)
        if n1_norm < 1e-8:
            return np.empty((0, 3), dtype=np.float64)
        n1 = n1 / n1_norm
        n2 = np.cross(seg_dir, n1)

        alphas = self.rng.uniform(0.05, 0.95, size=int(n_points))
        radial_r = np.sqrt(self.rng.uniform(0.0, 1.0, size=int(n_points))) * float(radius)
        radial_theta = self.rng.uniform(0.0, 2.0 * np.pi, size=int(n_points))

        base_points = start_pos[None, :] + alphas[:, None] * seg[None, :]
        offsets = (
            np.cos(radial_theta)[:, None] * radial_r[:, None] * n1[None, :]
            + np.sin(radial_theta)[:, None] * radial_r[:, None] * n2[None, :]
        )
        points = base_points + offsets
        points = np.clip(points, self.planner.bounds_min, self.planner.bounds_max)
        return points.astype(np.float64)

    @staticmethod
    def _exclude_aabb_near_points(obs_aabb, points, radius):
        if obs_aabb is None:
            return None

        obs_aabb = np.asarray(obs_aabb, dtype=np.float64)
        if obs_aabb.ndim != 2 or obs_aabb.shape[1] != 6 or obs_aabb.shape[0] == 0:
            return None

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[0] == 0:
            return obs_aabb

        radius = float(max(radius, 0.0))
        mins = obs_aabb[:, :3]
        maxs = mins + obs_aabb[:, 3:6]

        keep = np.ones(obs_aabb.shape[0], dtype=bool)
        for pnt in points:
            closest = np.minimum(np.maximum(pnt[None, :], mins), maxs)
            dist = np.linalg.norm(closest - pnt[None, :], axis=1)
            keep &= dist > radius

        filtered = obs_aabb[keep]
        if filtered.shape[0] == 0:
            return None
        return filtered

    def _should_reset_tree(self, current_pos, goal_pos, current_t):
        if self._active_tree_nodes is None or len(self._active_tree_nodes) == 0:
            return True

        if self._active_tree_start_pos is None or self._active_tree_goal_pos is None:
            return True

        if np.linalg.norm(np.asarray(goal_pos) - self._active_tree_goal_pos) > 0.02:
            return True

        if np.linalg.norm(np.asarray(current_pos) - self._active_tree_start_pos) > self.tree_resume_pos_tol:
            return True

        if float(current_t) < float(self._active_tree_start_t):
            return True

        if (float(current_t) - float(self._active_tree_start_t)) > self.tree_resume_time_window:
            return True

        return False

    @staticmethod
    def _trim_path_prefix(path, current_pos, keep_distance=0.02):
        if path is None or len(path) == 0:
            return []

        current_pos = np.asarray(current_pos, dtype=np.float64)
        keep_distance = float(max(keep_distance, 1e-4))

        first_keep = None
        for idx, (pos, _) in enumerate(path):
            if np.linalg.norm(np.asarray(pos, dtype=np.float64) - current_pos) > keep_distance:
                first_keep = idx
                break

        if first_keep is None:
            return []
        return list(path[first_keep:])

    def needs_replan(self, current_pos, current_t):
        if len(self.current_path) == 0:
            return True

        return not self.planner.remaining_path_valid(
            self.current_path,
            current_pos=current_pos,
            current_t=current_t,
            dynamic_obstacle_fn=self.dynamic_obstacle_fn,
            arm_collision_fn=self._arm_collision_fn,
        )

    def replan(self, point_cloud, current_pos, goal_pos, current_t, visualize=False, static_obs_aabb=None, iter_budget=None):
        self.stats["replan_requests"] += 1
        self._update_stuck_state(current_pos=current_pos, goal_pos=goal_pos, current_t=current_t)
        previous_path = list(self.current_path)

        if static_obs_aabb is not None:
            safe_obs_aabb = self._exclude_aabb_near_points(
                static_obs_aabb,
                np.vstack([
                    np.asarray(current_pos, dtype=np.float64),
                    np.asarray(goal_pos, dtype=np.float64),
                ]),
                radius=max(0.06, self.planner.collision_margin * 2.0),
            )
            self.planner.set_static_aabb_obstacles(safe_obs_aabb)

        iter_budget = self.incremental_iter_budget if iter_budget is None else int(max(1, iter_budget))

        tree_reset = self._should_reset_tree(current_pos, goal_pos, current_t)
        if tree_reset:
            start_valid = self.planner._is_state_valid(
                np.asarray(current_pos, dtype=np.float64),
                float(current_t),
                dynamic_obstacle_fn=self.dynamic_obstacle_fn,
                arm_collision_fn=self._arm_collision_fn,
            )

            if (not start_valid) and (self.planner.static_obs_aabb is not None):
                # Static map can include robot/self points near EE. Relax locally before giving up.
                relaxed_obs = self._exclude_aabb_near_points(
                    self.planner.static_obs_aabb,
                    np.asarray(current_pos, dtype=np.float64),
                    radius=max(0.10, self.planner.collision_margin * 2.5),
                )
                self.planner.set_static_aabb_obstacles(relaxed_obs)
                start_valid = self.planner._is_state_valid(
                    np.asarray(current_pos, dtype=np.float64),
                    float(current_t),
                    dynamic_obstacle_fn=self.dynamic_obstacle_fn,
                    arm_collision_fn=self._arm_collision_fn,
                )

            if not start_valid:
                self.stats["plan_failures"] += 1
                return {
                    "heuristic_ok": False,
                    "num_runs": 0,
                    "num_heuristic_points": 0,
                    "plan_ok": False,
                    "kept_previous_path": len(previous_path) > 0,
                    "tree_reset": True,
                    "tree_size": 0,
                    "iter_budget": int(iter_budget),
                }
            self._active_tree_nodes = self.planner.init_tree(current_pos, current_t)
            self._active_tree_start_pos = np.asarray(current_pos, dtype=np.float64).copy()
            self._active_tree_start_t = float(current_t)
            self._active_tree_goal_pos = np.asarray(goal_pos, dtype=np.float64).copy()
            self._active_subtrees = []

        fallback_corridor_used = False
        fallback_corridor_points = 0

        heuristic_refreshed = False
        should_refresh_heuristic = (
            tree_reset
            or (self.planner.heuristic_points is None)
            or ((float(current_t) - float(self._last_heuristic_update_t)) >= self.heuristic_update_interval and point_cloud is not None)
        )

        if should_refresh_heuristic and point_cloud is not None and point_cloud.shape[0] > 0 and self.predictor is not None:
            self.stats["prediction_attempts"] += 1
            success, heuristic_points, _, num_runs = self.predictor.predict(
                pc=point_cloud,
                x_start=current_pos,
                x_goal=goal_pos,
                neighbor_radius=self.neighbor_radius,
                max_trial_attempts=self.max_trial_attempts,
                visualize=visualize,
            )
            self.stats["total_num_runs"] += int(num_runs)
            if success:
                self.planner.set_heuristic_points(heuristic_points)
                self.stats["heuristic_successes"] += 1
                self.stats["total_heuristic_points"] += int(heuristic_points.shape[0])
                self._last_heuristic_update_t = float(current_t)
                heuristic_refreshed = True
                status = {
                    "heuristic_ok": True,
                    "num_runs": int(num_runs),
                    "num_heuristic_points": int(heuristic_points.shape[0]),
                }
            else:
                corridor_points = self._build_corridor_heuristic_points(current_pos, goal_pos)
                if corridor_points.shape[0] > 0:
                    self.planner.set_heuristic_points(corridor_points)
                    fallback_corridor_used = True
                    fallback_corridor_points = int(corridor_points.shape[0])
                else:
                    self.planner.set_heuristic_points(None)
                self.stats["heuristic_failures"] += 1
                self.stats["fallback_count"] += 1
                self._last_heuristic_update_t = float(current_t)
                heuristic_refreshed = True
                status = {
                    "heuristic_ok": False,
                    "num_runs": int(num_runs),
                    "num_heuristic_points": 0,
                    "fallback_corridor_used": fallback_corridor_used,
                    "fallback_corridor_points": fallback_corridor_points,
                }
        elif should_refresh_heuristic:
            corridor_points = self._build_corridor_heuristic_points(current_pos, goal_pos)
            if corridor_points.shape[0] > 0:
                self.planner.set_heuristic_points(corridor_points)
                fallback_corridor_used = True
                fallback_corridor_points = int(corridor_points.shape[0])
            else:
                self.planner.set_heuristic_points(None)
            self.stats["fallback_count"] += 1
            self._last_heuristic_update_t = float(current_t)
            heuristic_refreshed = True
            status = {
                "heuristic_ok": False,
                "num_runs": 0,
                "num_heuristic_points": 0,
                "fallback_corridor_used": fallback_corridor_used,
                "fallback_corridor_points": fallback_corridor_points,
            }
        else:
            status = {
                "heuristic_ok": self.planner.heuristic_points is not None,
                "num_runs": 0,
                "num_heuristic_points": 0 if self.planner.heuristic_points is None else int(self.planner.heuristic_points.shape[0]),
            }

        self._active_tree_nodes, self._active_subtrees, new_path = self.planner.plan_incremental(
            start=self._active_tree_start_pos,
            goal=goal_pos,
            start_time=self._active_tree_start_t,
            tree_nodes=self._active_tree_nodes,
            subtrees=self._active_subtrees,
            iter_budget=iter_budget,
            dynamic_obstacle_fn=self.dynamic_obstacle_fn,
            arm_collision_fn=self._arm_collision_fn,
            arm_singularity_fn=self._arm_singularity_fn,
            rng=self.rng,
        )

        status["tree_reset"] = tree_reset
        status["tree_size"] = 0 if self._active_tree_nodes is None else int(len(self._active_tree_nodes))
        status["subtree_count"] = int(len(self._active_subtrees))
        self.stats["subtree_expansions"] += int(len(self._active_subtrees))
        status["iter_budget"] = int(iter_budget)
        status["heuristic_refreshed"] = bool(heuristic_refreshed)
        status["adaptive_heuristic_bias"] = float(self._adaptive_heuristic_bias)

        if new_path is None:
            if len(previous_path) > 0:
                self.current_path = previous_path
                status["kept_previous_path"] = True
            else:
                self.current_path = []
                status["kept_previous_path"] = False
            self.stats["plan_failures"] += 1
            self._consecutive_plan_failures += 1
            self._maybe_decay_bias()
            status["plan_ok"] = False
            status["plan_in_progress"] = True
            return status

        trimmed_path = self._trim_path_prefix(new_path, current_pos=current_pos, keep_distance=0.02)
        smoothed_path = self._shortcut_smooth_path(trimmed_path, current_t=current_t)
        if len(smoothed_path) < len(trimmed_path):
            self.stats["smoothing_applied"] += 1
        self.current_path = smoothed_path
        self.stats["plan_successes"] += 1
        self.stats["total_waypoints_generated"] += len(self.current_path)
        self._consecutive_plan_failures = 0
        self._recover_bias_on_success()
        status["plan_ok"] = True
        status["plan_in_progress"] = False
        status["num_waypoints"] = len(self.current_path)
        return status

    def next_waypoint(self):
        if len(self.current_path) == 0:
            return None
        return self.current_path.pop(0)

    def get_stats_snapshot(self):
        stats = dict(self.stats)
        prediction_attempts = max(1, stats["prediction_attempts"])
        stats["heuristic_success_rate"] = stats["heuristic_successes"] / prediction_attempts
        stats["avg_heuristic_points"] = stats["total_heuristic_points"] / prediction_attempts
        stats["avg_png_runs"] = stats["total_num_runs"] / prediction_attempts
        return stats


class AtomicPathUpdate:
    """Thread-safe latest-path container for lock-protected atomic swaps."""

    def __init__(self):
        self._lock = threading.Lock()
        self._version = 0
        self._path = []
        self._status = {}

    def publish(self, path, status):
        safe_path = [] if path is None else list(path)
        safe_status = {} if status is None else dict(status)
        with self._lock:
            self._path = safe_path
            self._status = safe_status
            self._version += 1
            return self._version

    def get_if_newer(self, last_version):
        with self._lock:
            if self._version <= int(last_version):
                return int(last_version), None, None
            return self._version, list(self._path), dict(self._status)


class AsyncPlanningNode:
    """Background planning node that runs predictor + RRT off the simulation thread."""

    def __init__(self, planning_manager, iter_budget=80, visualize=False):
        self.planning_manager = planning_manager
        self.iter_budget = int(max(1, iter_budget))
        self.visualize = bool(visualize)

        self.result_store = AtomicPathUpdate()

        self._request_lock = threading.Lock()
        self._latest_request = None
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="AsyncPlanningNode", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._request_event.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(
        self,
        point_cloud,
        static_obs_aabb,
        current_pos,
        goal_pos,
        current_t,
    ):
        request = {
            "point_cloud": None if point_cloud is None else np.asarray(point_cloud, dtype=np.float64).copy(),
            "static_obs_aabb": None if static_obs_aabb is None else np.asarray(static_obs_aabb, dtype=np.float64).copy(),
            "current_pos": np.asarray(current_pos, dtype=np.float64).copy(),
            "goal_pos": np.asarray(goal_pos, dtype=np.float64).copy(),
            "current_t": float(current_t),
            "iter_budget": int(self.iter_budget),
        }
        with self._request_lock:
            # Keep only the latest request to avoid backlog under high-frequency submission.
            self._latest_request = request
            self._request_event.set()

    def _pop_latest_request(self):
        with self._request_lock:
            req = self._latest_request
            self._latest_request = None
            if self._latest_request is None:
                self._request_event.clear()
            return req

    def _run_loop(self):
        while not self._stop_event.is_set():
            self._request_event.wait(timeout=0.05)
            if self._stop_event.is_set():
                break

            req = self._pop_latest_request()
            if req is None:
                continue

            t0 = time.perf_counter()
            status = self.planning_manager.replan(
                point_cloud=req["point_cloud"],
                current_pos=req["current_pos"],
                goal_pos=req["goal_pos"],
                current_t=req["current_t"],
                visualize=self.visualize,
                static_obs_aabb=req["static_obs_aabb"],
                iter_budget=req["iter_budget"],
            )
            status = {} if status is None else dict(status)
            status["planning_wall_time_ms"] = (time.perf_counter() - t0) * 1000.0
            path = list(self.planning_manager.current_path)
            self.result_store.publish(path=path, status=status)




