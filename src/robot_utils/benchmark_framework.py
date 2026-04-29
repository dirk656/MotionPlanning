import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence
from robot_utils.benchmark_scenrio import BasePlanner, PlannerOutput, Scenario
from robot_utils.rrt_base_3d import SimpleRRT
from robot_utils.rrt_informed import InformedRRTStar
from robot_utils.rtt_star import RRTStar

import numpy as np

class RRTWrapper(BasePlanner):
    def __init__(self, rrt, name="rrt"):
        self.rrt = rrt
        self.name = str(name)

    def plan(self, scenario, current_t=0.0):
        path = self.rrt.plan(scenario.start, scenario.goal)
        success = path is not None and len(path) > 0
        return PlannerOutput(path=path, success=success, planning_time=0.0, extra={"planner": self.name, "replan_count": 1})


class HeuristicPlannerWrapper(BasePlanner):
    def __init__(self, manager, use_heuristic=True, use_subtree=True, use_risk=True):
        self.manager = manager
        self.use_heuristic = bool(use_heuristic)
        self.use_subtree = bool(use_subtree)
        self.use_risk = bool(use_risk)
        self.base_predictor = manager.predictor
        self.base_bias = float(manager.planner.base_heuristic_bias)
        self._apply_ablation_controls()

    def apply_ablation_controls(self):
        if not self.use_subtree:
            self.manager.planner.max_subtrees = 0
            self.manager.planner.subtree_spawn_prob = 0.0
        if not self.use_risk:
            self.manager.planner.risk_lookahead_s = 0.0

        if not self.use_heuristic:
            self.manager.predictor = None
            self.manager.planner.set_heuristic_points(None)
            self.manager.adaptive_heuristic_bias = 0.0
            self.manager.planner.set_runtime_heuristic_bias(0.0)
        else:
            self.manager.predictor = self.base_predictor
            self.manager.adaptive_heuristic_bias = self.base_bias
            self.manager.planner.set_runtime_heuristic_bias(self.base_bias)

    def plan(self, scenario, current_t=0.0):
        self.apply_ablation_controls()
        status = self.manager.replan(
            point_cloud=scenario.point_cloud,
            current_pos=scenario.start,
            goal_pos=scenario.goal,
            current_t=current_t,
            static_obs_aabb=scenario.static_obs,
        )
        path = list(self.manager.current_path)
        status = {} if status is None else dict(status)
        success = bool(status.get("plan_ok", False)) if status else (path is not None and len(path) > 0)
        status.setdefault("planner", "heuristic")
        status.setdefault("replan_count", 1)
        status.setdefault("heuristic_bias", float(self.manager.adaptive_heuristic_bias))
        status.setdefault("heuristic_points", 0 if self.manager.planner.heuristic_points is None else int(self.manager.planner.heuristic_points.shape[0]))
        status.setdefault("heuristic_usage_ratio", 1.0 if self.use_heuristic else 0.0)
        return PlannerOutput(path=path, success=success, planning_time=0.0, extra=status)

    def is_path_valid(self, path, scenario, current_t=0.0):
        return self.manager.planner.remaining_path_valid(
            path,
            scenario.start,
            current_t,
            dynamic_obstacle_fn=scenario.dynamic_fn,
        )

    def reset(self, seed=None):
        self.manager.current_path = []
        self.manager.active_tree_nodes = None
        self.manager.active_tree_start_pos = None
        self.manager.active_tree_start_t = None
        self.manager.active_tree_goal_pos = None
        self.manager.active_subtrees = []
        self.manager.last_heuristic_update_t = -1e9
        self.manager.consecutive_plan_failures = 0
        self.manager.best_goal_dist = float("inf")
        self.manager.last_progress_t = -1e9
        self.manager.adaptive_heuristic_bias = float(self.manager.planner.base_heuristic_bias)
        self.manager.planner.set_runtime_heuristic_bias(self.manager.adaptive_heuristic_bias)
        self.manager.stats = {k: 0 for k in self.manager.stats}
        if seed is not None:
            self.manager.rng = np.random.default_rng(int(seed))


def build_planner(config):
    planner_name = str(config.get("planner", "rrt")).lower()
    if planner_name == "rrt":
        rrt_cfg = config.get("rrt", {})
        rrt = SimpleRRT(
            config["bounds_min"],
            config["bounds_max"],
            step_len=rrt_cfg.get("step_len", 0.1),
            max_iters=rrt_cfg.get("max_iters", 1000),
            goal_tolerance=rrt_cfg.get("goal_tolerance", 0.1),
        )
        return RRTWrapper(rrt, name="rrt")

    if planner_name in ("rrtstar", "rrt_star"):
        rrt_cfg = config.get("rrtstar", {})
        rrt = RRTStar(
            config["bounds_min"],
            config["bounds_max"],
            step_len=rrt_cfg.get("step_len", 0.1),
            max_iters=rrt_cfg.get("max_iters", 1000),
            goal_tolerance=rrt_cfg.get("goal_tolerance", 0.1),
            radius=rrt_cfg.get("radius", 0.2),
        )
        return RRTWrapper(rrt, name="rrtstar")

    if planner_name in ("informed_rrtstar", "informed_rrt_star"):
        rrt_cfg = config.get("informed_rrtstar", {})
        rrt = InformedRRTStar(
            config["bounds_min"],
            config["bounds_max"],
            step_len=rrt_cfg.get("step_len", 0.1),
            max_iters=rrt_cfg.get("max_iters", 1000),
            goal_tolerance=rrt_cfg.get("goal_tolerance", 0.1),
            radius=rrt_cfg.get("radius", 0.2),
            sample_attempts=rrt_cfg.get("sample_attempts", 30),
        )
        return RRTWrapper(rrt, name="informed_rrtstar")

    if planner_name == "heuristic":
        manager = config.get("manager")
        if manager is None:
            raise ValueError("Heuristic planner requires a manager instance")
        return HeuristicPlannerWrapper(
            manager,
            use_heuristic=config.get("use_heuristic", True),
            use_subtree=config.get("use_subtree", True),
            use_risk=config.get("use_risk", True),
        )

    raise ValueError(f"Unknown planner: {planner_name}")


class BenchmarkRunner:
    def __init__(self, planners, scenarios, num_runs=10, seed_offset=0, online_mode=False, time_bins=None):
        self.planners = planners if isinstance(planners, dict) else {"planner": planners}
        self.scenarios = list(scenarios)
        self.num_runs = int(max(1, num_runs))
        self.seed_offset = int(seed_offset)
        self.online_mode = bool(online_mode)
        self.time_bins = list(time_bins) if time_bins is not None else [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    def run(self):
        results = {}
        for name, planner_entry in self.planners.items():
            planner_results = []
            for s_idx, scenario in enumerate(self.scenarios):
                metrics = []
                for run_idx in range(self.num_runs):
                    seed = self.seed_offset + run_idx
                    np.random.seed(seed)

                    planner = planner_entry() if callable(planner_entry) else planner_entry
                    if hasattr(planner, "reset"):
                        planner.reset(seed=seed)

                    if self.online_mode:
                        metrics.append(self._run_online(planner, scenario))
                    else:
                        t0 = time.perf_counter()
                        output = planner.plan(scenario, current_t=0.0)
                        planning_time = time.perf_counter() - t0

                        if hasattr(output, "planning_time"):
                            output.planning_time = float(planning_time)

                        metrics.append(self._evaluate(output, scenario, planning_time))

                planner_results.append(
                    {
                        "scenario": scenario.name or f"scenario_{s_idx}",
                        "aggregate": self._aggregate(metrics),
                        "runs": metrics,
                    }
                )
            results[name] = planner_results
        return results

    def evaluate(self, output, scenario, planning_time):
        if output is None:
            return {
                "success": 0,
                "planning_time": float(planning_time),
                "path_length": float("nan"),
                "collision_rate": float("nan"),
                "min_clearance": float("nan"),
                "replan_count": 0,
                "time_to_first_solution": float("nan"),
                "replan_latency": float("nan"),
                "tracking_error": float("nan"),
            }

        points = self._normalize_path(output.path)
        success = int(bool(output.success))
        if not success or len(points) == 0:
            return {
                "success": 0,
                "planning_time": float(planning_time),
                "path_length": float("nan"),
                "collision_rate": float("nan"),
                "min_clearance": float("nan"),
                "replan_count": int(output.extra.get("replan_count", 0)),
                "time_to_first_solution": float("nan"),
                "replan_latency": float("nan"),
                "tracking_error": float("nan"),
            }

        path_length = self._compute_path_length(points, scenario.start)
        times = self._compute_times(points, scenario.start, 0.0, scenario.speed)
        collision_rate, min_clearance = self._compute_clearance_stats(points, times, scenario)
        replan_count = int(output.extra.get("replan_count", 1))

        return {
            "success": 1,
            "planning_time": float(planning_time),
            "path_length": float(path_length),
            "collision_rate": float(collision_rate) if collision_rate is not None else float("nan"),
            "min_clearance": float(min_clearance) if min_clearance is not None else float("nan"),
            "replan_count": replan_count,
            "time_to_first_solution": float(planning_time),
            "replan_latency": float("nan"),
            "tracking_error": float("nan"),
        }

    def aggregate(self, metrics):
        success_vals = [m["success"] for m in metrics]
        length_vals = [m["path_length"] for m in metrics if np.isfinite(m["path_length"])]
        time_vals = [m["planning_time"] for m in metrics if np.isfinite(m["planning_time"])]
        collision_vals = [m["collision_rate"] for m in metrics if np.isfinite(m["collision_rate"])]
        clearance_vals = [m["min_clearance"] for m in metrics if np.isfinite(m["min_clearance"])]
        replan_vals = [m["replan_count"] for m in metrics if np.isfinite(m["replan_count"])]
        first_solution_vals = [m["time_to_first_solution"] for m in metrics if np.isfinite(m["time_to_first_solution"])]
        replan_latency_vals = [m["replan_latency"] for m in metrics if np.isfinite(m["replan_latency"])]
        tracking_err_vals = [m["tracking_error"] for m in metrics if np.isfinite(m["tracking_error"])]

        return {
            "success_rate": float(np.mean(success_vals)) if success_vals else float("nan"),
            "path_length_mean": float(np.mean(length_vals)) if length_vals else float("nan"),
            "path_length_std": float(np.std(length_vals)) if length_vals else float("nan"),
            "planning_time_mean": float(np.mean(time_vals)) if time_vals else float("nan"),
            "planning_time_std": float(np.std(time_vals)) if time_vals else float("nan"),
            "collision_rate_mean": float(np.mean(collision_vals)) if collision_vals else float("nan"),
            "min_clearance_mean": float(np.mean(clearance_vals)) if clearance_vals else float("nan"),
            "replan_count_mean": float(np.mean(replan_vals)) if replan_vals else float("nan"),
            "time_to_first_solution_mean": float(np.mean(first_solution_vals)) if first_solution_vals else float("nan"),
            "time_to_first_solution_std": float(np.std(first_solution_vals)) if first_solution_vals else float("nan"),
            "replan_latency_mean": float(np.mean(replan_latency_vals)) if replan_latency_vals else float("nan"),
            "tracking_error_mean": float(np.mean(tracking_err_vals)) if tracking_err_vals else float("nan"),
            "success_rate_curve": self._build_success_curve(first_solution_vals),
            "path_length_distribution": list(length_vals),
            "planning_time_distribution": list(time_vals),
        }

    def run_online(self, planner, scenario):
        current_pos = np.asarray(scenario.start, dtype=np.float64).copy()
        current_t = 0.0
        path_points = []
        path_idx = 0
        executed_points = [current_pos.copy()]
        executed_times = [current_t]
        replan_count = 0
        planning_times = []
        first_solution_time = float("nan")

        goal_tol = float(scenario.goal_tolerance)
        sim_dt = float(scenario.sim_dt)
        max_sim_time = float(scenario.max_sim_time)

        while current_t <= max_sim_time:
            if not path_points or path_idx >= len(path_points) or not self._path_valid(path_points[path_idx:], current_pos, current_t, scenario):
                t0 = time.perf_counter()
                output = planner.replan(scenario, current_t=current_t)
                planning_time = time.perf_counter() - t0
                planning_times.append(float(planning_time))
                replan_count += 1

                if output is None or not output.success:
                    return {
                        "success": 0,
                        "planning_time": float(np.sum(planning_times)) if planning_times else float(planning_time),
                        "path_length": float("nan"),
                        "collision_rate": float("nan"),
                        "min_clearance": float("nan"),
                        "replan_count": int(replan_count),
                        "time_to_first_solution": first_solution_time,
                        "replan_latency": float(np.mean(planning_times)) if planning_times else float("nan"),
                        "tracking_error": float("nan"),
                    }

                if not np.isfinite(first_solution_time):
                    first_solution_time = float(planning_time)

                path_points = self._normalize_path(output.path)
                path_idx = 0

            if path_idx < len(path_points):
                next_point = np.asarray(path_points[path_idx], dtype=np.float64)
                current_pos = next_point
                path_idx += 1
            current_t += sim_dt
            executed_points.append(current_pos.copy())
            executed_times.append(current_t)

            if np.linalg.norm(current_pos - scenario.goal) <= goal_tol:
                break

        success = np.linalg.norm(current_pos - scenario.goal) <= goal_tol
        path_length = self._compute_path_length(executed_points[1:], executed_points[0])
        collision_rate, min_clearance = self._compute_clearance_stats(executed_points, executed_times, scenario)
        total_planning_time = float(np.sum(planning_times)) if planning_times else float("nan")
        replan_latency = float(np.mean(planning_times)) if planning_times else float("nan")

        return {
            "success": int(bool(success)),
            "planning_time": total_planning_time,
            "path_length": float(path_length) if success else float("nan"),
            "collision_rate": float(collision_rate) if collision_rate is not None else float("nan"),
            "min_clearance": float(min_clearance) if min_clearance is not None else float("nan"),
            "replan_count": int(replan_count),
            "time_to_first_solution": float(first_solution_time),
            "replan_latency": float(replan_latency),
            "tracking_error": float("nan"),
        }

    def path_valid(self, points, current_pos, current_t, scenario):
        if points is None or len(points) == 0:
            return False
        path = [np.asarray(current_pos, dtype=np.float64)] + [np.asarray(p, dtype=np.float64) for p in points]
        times = self._compute_times(path, current_pos, current_t, scenario.speed)
        collision_rate, _ = self._compute_clearance_stats(path, times, scenario)
        return collision_rate is None or collision_rate <= 0.0

    def build_success_curve(self, times):
        if not times:
            return []
        curve = []
        valid_times = [float(t) for t in times if np.isfinite(t)]
        if not valid_times:
            return []
        for t_bin in self.time_bins:
            rate = float(np.mean([1.0 if t <= t_bin else 0.0 for t in valid_times]))
            curve.append({"time": float(t_bin), "success_rate": rate})
        return curve

    @staticmethod
    def normalize_path(path):
        if path is None:
            return []
        points = []
        for p in path:
            if isinstance(p, (tuple, list)) and len(p) == 2:
                p = p[0]
            points.append(np.asarray(p, dtype=np.float64))
        return points

    @staticmethod
    def compute_path_length(points, start):
        if not points:
            return float("nan")
        length = 0.0
        prev = np.asarray(start, dtype=np.float64)
        for p in points:
            length += float(np.linalg.norm(p - prev))
            prev = p
        return float(length)

    @staticmethod
    def compute_times(points, start, base_t, speed):
        speed = max(float(speed), 1e-6)
        times = []
        dist_acc = 0.0
        prev = np.asarray(start, dtype=np.float64)
        for p in points:
            dist_acc += float(np.linalg.norm(p - prev))
            times.append(float(base_t + dist_acc / speed))
            prev = p
        return times

    def compute_clearance_stats(self, points, times, scenario):
        static_obs = scenario.static_obs
        dynamic_fn = scenario.dynamic_fn

        if static_obs is None and dynamic_fn is None:
            return None, None

        if static_obs is not None and static_obs.ndim == 1:
            static_obs = static_obs[None, :]

        total = 0
        collisions = 0
        min_clearance = float("inf")

        for idx, p in enumerate(points):
            t_abs = float(times[idx]) if idx < len(times) else 0.0
            clearance = float("inf")

            if static_obs is not None:
                for aabb in static_obs:
                    clearance = min(clearance, self._aabb_clearance(p, aabb, scenario.static_obs_format))

            if dynamic_fn is not None:
                for obs in dynamic_fn(t_abs) or []:
                    center, radius = self._parse_sphere(obs)
                    if center is None:
                        continue
                    clearance = min(clearance, float(np.linalg.norm(p - center) - radius))

            if clearance < float("inf"):
                total += 1
                min_clearance = min(min_clearance, clearance)
                if clearance <= 0.0:
                    collisions += 1

        if total == 0:
            return None, None

        return float(collisions) / float(total), min_clearance

    @staticmethod
    def aabb_clearance(point, aabb, fmt):
        mins = np.asarray(aabb[:3], dtype=np.float64)
        if str(fmt).lower() == "minmax":
            maxs = np.asarray(aabb[3:6], dtype=np.float64)
        else:
            maxs = mins + np.asarray(aabb[3:6], dtype=np.float64)
        closest = np.minimum(np.maximum(point, mins), maxs)
        return float(np.linalg.norm(point - closest))

    @staticmethod
    def parse_sphere(obs):
        if hasattr(obs, "center") and hasattr(obs, "radius"):
            return np.asarray(obs.center, dtype=np.float64), float(obs.radius)
        if isinstance(obs, dict):
            center = obs.get("center")
            radius = obs.get("radius")
            if center is None or radius is None:
                return None, None
            return np.asarray(center, dtype=np.float64), float(radius)
        if isinstance(obs, (tuple, list)) and len(obs) >= 2:
            return np.asarray(obs[0], dtype=np.float64), float(obs[1])
        return None, None


def build_default_scenarios(bounds_min, bounds_max, start, goal, config=None, rng=None):
    cfg = {} if config is None else dict(config)
    rng = np.random.default_rng() if rng is None else rng

    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    scenario_types = cfg.get(
        "scenario_types",
        ["easy_static", "cluttered_static", "dynamic_crossing", "human_interaction"],
    )
    num_variants = int(cfg.get("num_variants", 1))
    obs_counts = cfg.get(
        "obstacle_counts",
        {"easy_static": 4, "cluttered_static": 14, "dynamic_crossing": 8, "human_interaction": 10},
    )
    obs_size = cfg.get("obstacle_size_range", [0.05, 0.12])
    obs_clearance = float(cfg.get("obstacle_clearance", 0.08))
    dynamic_speed = cfg.get(
        "dynamic_speed",
        {"dynamic_crossing": 0.4, "human_interaction": 0.6},
    )
    dynamic_radius = float(cfg.get("dynamic_radius", 0.08))
    static_obs_format = str(cfg.get("static_obs_format", "min_size"))
    speed = float(cfg.get("speed", 1.0))
    goal_tolerance = float(cfg.get("goal_tolerance", 0.05))
    sim_dt = float(cfg.get("sim_dt", 0.1))
    max_sim_time = float(cfg.get("max_sim_time", 12.0))

    scenarios = []
    for name in scenario_types:
        base_count = int(obs_counts.get(name, 6))
        base_speed = float(dynamic_speed.get(name, 0.5))
        for v in range(num_variants):
            count = max(0, base_count + int(rng.integers(-2, 3)))
            static_obs = _generate_static_aabbs(
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                count=count,
                size_range=obs_size,
                start=start,
                goal=goal,
                clearance=obs_clearance,
                rng=rng,
            )
            dynamic_fn = None
            if name in ("dynamic_crossing", "human_interaction"):
                dynamic_fn = build_dynamic_fn(
                    name=name,
                    start=start,
                    goal=goal,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    speed=base_speed * float(rng.uniform(0.8, 1.2)),
                    radius=dynamic_radius,
                )

            scenarios.append(
                Scenario(
                    start=start,
                    goal=goal,
                    static_obs=static_obs,
                    dynamic_fn=dynamic_fn,
                    point_cloud=None,
                    name=f"{name}_v{v}",
                    speed=speed,
                    static_obs_format=static_obs_format,
                    goal_tolerance=goal_tolerance,
                    sim_dt=sim_dt,
                    max_sim_time=max_sim_time,
                )
            )

    return scenarios


def generate_static_aabbs(bounds_min, bounds_max, count, size_range, start, goal, clearance, rng):
    size_min = float(size_range[0])
    size_max = float(size_range[1])
    obstacles = []
    max_tries = max(100, count * 50)

    for _ in range(max_tries):
        if len(obstacles) >= count:
            break
        size = rng.uniform(size_min, size_max, size=3)
        mins = rng.uniform(bounds_min, bounds_max - size)
        aabb = np.hstack([mins, size])
        if aabb_clearance_point(start, aabb) < clearance:
            continue
        if aabb_clearance_point(goal, aabb) < clearance:
            continue
        obstacles.append(aabb)

    if len(obstacles) == 0:
        return None
    return np.asarray(obstacles, dtype=np.float64)


def aabb_clearance_point(point, aabb):
    mins = np.asarray(aabb[:3], dtype=np.float64)
    maxs = mins + np.asarray(aabb[3:6], dtype=np.float64)
    closest = np.minimum(np.maximum(point, mins), maxs)
    return float(np.linalg.norm(point - closest))


def build_dynamic_fn(name, start, goal, bounds_min, bounds_max, speed, radius):
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)

    seg = goal - start
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6:
        seg = np.array([1.0, 0.0, 0.0])
    seg_dir = seg / max(seg_len, 1e-6)
    basis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(seg_dir, basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perp = np.cross(seg_dir, basis)
    perp = perp / max(np.linalg.norm(perp), 1e-6)

    mid = 0.5 * (start + goal)
    offset = 0.25 * seg_len * perp

    if name == "human_interaction":
        p0 = np.clip(mid - offset, bounds_min, bounds_max)
        p1 = np.clip(mid + offset, bounds_min, bounds_max)
        p2 = np.clip(start + offset * 0.6, bounds_min, bounds_max)
        p3 = np.clip(goal - offset * 0.6, bounds_min, bounds_max)
        return make_pingpong_multi_fn([(p0, p1), (p2, p3)], speed=speed, radius=radius)

    p0 = np.clip(mid - offset, bounds_min, bounds_max)
    p1 = np.clip(mid + offset, bounds_min, bounds_max)
    return make_pingpong_multi_fn([(p0, p1)], speed=speed, radius=radius)


def make_pingpong_multi_fn(segments, speed, radius):
    speed = max(float(speed), 1e-6)
    seg_data = []
    for p0, p1 in segments:
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        dist = float(np.linalg.norm(p1 - p0))
        period = max(dist / speed * 2.0, 1e-3)
        seg_data.append((p0, p1, period))

    def dynamic_fn(t_abs):
        obstacles = []
        t_abs = float(t_abs)
        for p0, p1, period in seg_data:
            phase = (t_abs % period) / period
            if phase <= 0.5:
                alpha = phase * 2.0
            else:
                alpha = 2.0 - phase * 2.0
            pos = p0 * (1.0 - alpha) + p1 * alpha
            obstacles.append({"center": pos, "radius": float(radius)})
        return obstacles

    return dynamic_fn
