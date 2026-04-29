import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pybullet as p
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.pybullet.env.env import (
    PyBulletMotionEnv,
    build_tube_motion,
    sample_reachable_start_goal,
    start_video_recording,
    stop_video_recording,
)
from src.pybullet.env.path_planner import (
    create_planning_bundle,
    create_predictor,
    poll_plan_update,
    poll_pointcloud_update,
    request_pointcloud_update,
    request_replan,
    start_planning_bundle,
    stop_planning_bundle,
)
from robot_utils.benchmark_framework import BenchmarkRunner, Scenario, build_default_scenarios, build_planner


from src.pybullet.env.path_tools import generate_pos
from src.pybullet.env.pointcloud_tools import load_workspace_bounds
from src.robot_utils.time_based_rrt import SphereObstacle





def parse_args():
    parser = argparse.ArgumentParser(description="Config-driven PyBullet online planning demo")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to main config yaml")
    parser.add_argument("--use_heuristic", type=bool, default=None, help="Enable/disable heuristic predictor")
    parser.add_argument("--ablation_group", type=str, default=None, help="Ablation tag: rrt_only / heuristic_only / full")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment seed")
    parser.add_argument("--max_wall_time_s", type=float, default=None, help="Override max run wall time")
    parser.add_argument("--exp_name", type=str, default=None, help="Optional fixed experiment folder name")
    parser.add_argument("--save_video", type=bool, default=None, help="Enable/disable mp4 recording")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return data   


def deep_update(base: Dict, patch: Dict):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def setup_experiment_folder(cfg: Dict, args):
    exp_cfg = cfg.get("experiment", {})
    root = Path(exp_cfg.get("output_root", "results/experiments"))
    root.mkdir(parents=True, exist_ok=True)

    if args.exp_name:
        exp_name = args.exp_name
    else:
        prefix = str(exp_cfg.get("name_prefix", "exp"))
        exp_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir



def setup_logger(exp_dir: Path, cfg: Dict):
    log_cfg = cfg.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("motionplanning")
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if bool(log_cfg.get("to_file", True)):
        file_handler = logging.FileHandler(exp_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def snapshot_code_and_config(exp_dir: Path, cfg: Dict, config_path: str):
    with open(exp_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    try:
        shutil.copy2(config_path, exp_dir / "config_input.yaml")
    except Exception:
        pass

    if not bool(cfg.get("experiment", {}).get("save_code_snapshot", True)):
        return

    snapshot_dir = exp_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        "src/pybullet/env/main.py",
        "src/pybullet/env/env.py",
        "src/pybullet/env/path_planner.py",
        "src/robot_utils/time_based_rrt.py",
        "src/robot_utils/predictor.py",
    ]
    for rel in files_to_copy:
        src = Path(rel)
        if src.exists():
            dst = snapshot_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def compute_path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    pts = [np.asarray(v[0], dtype=np.float64) for v in path]
    return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))


def compute_path_smoothness(path):
    if path is None or len(path) < 3:
        return 0.0
    pts = [np.asarray(v[0], dtype=np.float64) for v in path]
    angles = []
    for i in range(len(pts) - 2):
        v1 = pts[i + 1] - pts[i]
        v2 = pts[i + 2] - pts[i + 1]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles.append(float(np.arccos(cos_theta)))
    if not angles:
        return 0.0
    return float(np.mean(np.abs(angles)))


class MetricsLogger:
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.csv_path = exp_dir / "metrics.csv"
        self.rows = 0
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sim_time_s",
                    "planning_wall_time_ms",
                    "plan_ok",
                    "heuristic_ok",
                    "heuristic_refreshed",
                    "path_length",
                    "smoothness",
                    "num_waypoints",
                    "tree_size",
                    "subtree_count",
                    "iter_budget",
                    "adaptive_heuristic_bias",
                ]
            )

    def log_planning(self, sim_time_s: float, status: Dict, path):
        status = {} if status is None else dict(status)
        row = [
            float(sim_time_s),
            float(status.get("planning_wall_time_ms", 0.0)),
            int(bool(status.get("plan_ok", False))),
            int(bool(status.get("heuristic_ok", False))),
            int(bool(status.get("heuristic_refreshed", False))),
            compute_path_length(path),
            compute_path_smoothness(path),
            int(status.get("num_waypoints", len(path) if path is not None else 0)),
            int(status.get("tree_size", 0)),
            int(status.get("subtree_count", 0)),
            int(status.get("iter_budget", 0)),
            float(status.get("adaptive_heuristic_bias", 0.0)),
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.rows += 1


def save_result_json(exp_dir: Path, result: Dict):
    with open(exp_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def resolve_planner_name(exp_cfg: Dict, bench_cfg: Dict):
    planner = None if bench_cfg is None else bench_cfg.get("planner")
    if planner is None:
        planner = exp_cfg.get("planner")
    if planner is not None:
        return str(planner).lower()
    return "heuristic" if bool(exp_cfg.get("use_heuristic", True)) else "rrt"


def resolve_start_goal(exp_cfg: Dict, sim_env: PyBulletMotionEnv):
    if bool(exp_cfg.get("use_fixed_start_goal", True)):
        start_pos = np.asarray(exp_cfg.get("fixed_start_pos", [0.56, 0.16, 0.44]), dtype=np.float64)
        goal_pos = np.asarray(exp_cfg.get("fixed_goal_pos", [0.4, -0.4, 0.72]), dtype=np.float64)
        q_s = sim_env.solve_ik_continuous(start_pos, None)
        q_g = sim_env.solve_ik_continuous(goal_pos, None)
        if q_s is None or q_g is None:
            raise RuntimeError("Fixed start/goal are not IK-reachable. Adjust config fixed_start_pos/fixed_goal_pos.")
        return start_pos, goal_pos

    return sample_reachable_start_goal(
        ik_solver=sim_env.solve_ik_continuous,
        get_ee_position=sim_env.get_end_effector_pos,
        reset_joint_positions=sim_env.reset_joint_positions,
        get_joint_positions=sim_env.get_joint_positions,
        generate_pos_fn=generate_pos,
        max_tries=60,
    )


def run_benchmark_mode(cfg: Dict, exp_cfg: Dict, env_cfg: Dict, obstacle_cfg: Dict, bounds_min, bounds_max, seed: int, logger, sim_env: PyBulletMotionEnv):
    bench_cfg = cfg.get("benchmark", {})
    planner_name = resolve_planner_name(exp_cfg, bench_cfg)
    rng = np.random.default_rng(seed)

    sim_env.connect()
    try:
        start_pos, goal_pos = resolve_start_goal(exp_cfg, sim_env)
    finally:
        sim_env.disconnect()

    obstacle_radius = float(obstacle_cfg.get("radius", 0.09))
    obstacle_center_fn = build_tube_motion(
        start_pos=start_pos,
        goal_pos=goal_pos,
        tube_radius=float(obstacle_cfg.get("tube_radius", 0.07)),
        endpoint_clearance=float(obstacle_cfg.get("endpoint_clearance", 0.18)),
    )

    def dynamic_obstacle_fn(t_abs):
        center = np.clip(obstacle_center_fn(t_abs), bounds_min, bounds_max)
        return [SphereObstacle(center=np.asarray(center, dtype=np.float64), radius=obstacle_radius)]

    planner_cfg = cfg.get("planning", {}).get("planner", {})
    scenario_mode = str(bench_cfg.get("scenario_mode", "multi")).lower()
    if scenario_mode == "single":
        scenarios = [
            Scenario(
                start=start_pos,
                goal=goal_pos,
                static_obs=None,
                dynamic_fn=dynamic_obstacle_fn,
                point_cloud=None,
                name=str(bench_cfg.get("scenario_name", "default")),
                speed=float(planner_cfg.get("speed", 1.0)),
                static_obs_format=str(bench_cfg.get("static_obs_format", "min_size")),
                goal_tolerance=float(bench_cfg.get("goal_tolerance", 0.05)),
                sim_dt=float(bench_cfg.get("sim_dt", 0.1)),
                max_sim_time=float(bench_cfg.get("max_sim_time", 12.0)),
            )
        ]
    else:
        scenario_cfg = {
            "scenario_types": bench_cfg.get("scenario_types"),
            "num_variants": bench_cfg.get("num_variants", 1),
            "obstacle_counts": bench_cfg.get("obstacle_counts"),
            "obstacle_size_range": bench_cfg.get("obstacle_size_range", [0.05, 0.12]),
            "obstacle_clearance": bench_cfg.get("obstacle_clearance", 0.08),
            "dynamic_speed": bench_cfg.get("dynamic_speed"),
            "dynamic_radius": bench_cfg.get("dynamic_radius", 0.08),
            "static_obs_format": bench_cfg.get("static_obs_format", "min_size"),
            "speed": float(planner_cfg.get("speed", 1.0)),
            "goal_tolerance": bench_cfg.get("goal_tolerance", 0.05),
            "sim_dt": bench_cfg.get("sim_dt", 0.1),
            "max_sim_time": bench_cfg.get("max_sim_time", 12.0),
        }
        scenarios = build_default_scenarios(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            start=start_pos,
            goal=goal_pos,
            config=scenario_cfg,
            rng=rng,
        )

    planner_names = bench_cfg.get("planners")
    if planner_names is None:
        planner_names = [planner_name]
    elif isinstance(planner_names, str):
        planner_names = [planner_names]

    use_heuristic_flag = bench_cfg.get("use_heuristic")
    if use_heuristic_flag is None:
        use_heuristic_flag = bool(exp_cfg.get("use_heuristic", True))
    use_subtree_flag = bool(bench_cfg.get("use_subtree", True))
    use_risk_flag = bool(bench_cfg.get("use_risk", True))

    def build_planners_for_flags(flag_use_heuristic, flag_use_subtree, flag_use_risk, planner_list):
        planners_local = {}
        bundles_local = []
        for local_name in planner_list:
            local_name = str(local_name).lower()
            planner_config = {
                "planner": local_name,
                "bounds_min": bounds_min,
                "bounds_max": bounds_max,
            }
            if local_name == "heuristic":
                predictor = create_predictor(
                    config=cfg,
                    use_heuristic=bool(exp_cfg.get("use_heuristic", True)),
                    rng_seed=seed + 17,
                    logger=logger,
                )
                bundle = create_planning_bundle(
                    config=cfg,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    dynamic_obstacle_fn=dynamic_obstacle_fn,
                    predictor=predictor,
                    rng=rng,
                )
                planner_config["manager"] = bundle.planning_manager
                planner_config["use_heuristic"] = bool(flag_use_heuristic)
                planner_config["use_subtree"] = bool(flag_use_subtree)
                planner_config["use_risk"] = bool(flag_use_risk)
                bundles_local.append(bundle)
            elif local_name in ("rrtstar", "rrt_star"):
                planner_config["rrtstar"] = bench_cfg.get("rrtstar", {})
            elif local_name in ("informed_rrtstar", "informed_rrt_star"):
                planner_config["informed_rrtstar"] = bench_cfg.get("informed_rrtstar", {})
            else:
                planner_config["rrt"] = bench_cfg.get("rrt", {})

            planners_local[local_name] = build_planner(planner_config)
        return planners_local, bundles_local

    planners, planning_bundles = build_planners_for_flags(
        use_heuristic_flag,
        use_subtree_flag,
        use_risk_flag,
        planner_names,
    )

    runner = BenchmarkRunner(
        planners=planners,
        scenarios=scenarios,
        num_runs=int(bench_cfg.get("num_runs", 1)),
        seed_offset=int(bench_cfg.get("seed_offset", 0)),
        online_mode=bool(bench_cfg.get("online_mode", False)),
        time_bins=bench_cfg.get("time_bins"),
    )
    bench_result = runner.run()

    for bundle in planning_bundles:
        stop_planning_bundle(bundle)

    ablation_result = None
    auto_ablation = bool(bench_cfg.get("auto_ablation", False))
    ablation_groups = bench_cfg.get("ablation_groups")
    if auto_ablation or ablation_groups:
        if ablation_groups is None:
            ablation_groups = {
                "full": {"use_heuristic": True, "use_subtree": True, "use_risk": True},
                "no_heuristic": {"use_heuristic": False, "use_subtree": True, "use_risk": True},
                "no_subtree": {"use_heuristic": True, "use_subtree": False, "use_risk": True},
                "no_risk": {"use_heuristic": True, "use_subtree": True, "use_risk": False},
            }
        ablation_planners = bench_cfg.get("ablation_planners", ["heuristic"])
        if isinstance(ablation_planners, str):
            ablation_planners = [ablation_planners]

        ablation_result = {}
        for group_name, flags in ablation_groups.items():
            group_planners, group_bundles = build_planners_for_flags(
                flags.get("use_heuristic", True),
                flags.get("use_subtree", True),
                flags.get("use_risk", True),
                ablation_planners,
            )
            group_runner = BenchmarkRunner(
                planners=group_planners,
                scenarios=scenarios,
                num_runs=int(bench_cfg.get("num_runs", 1)),
                seed_offset=int(bench_cfg.get("seed_offset", 0)),
                online_mode=bool(bench_cfg.get("online_mode", False)),
                time_bins=bench_cfg.get("time_bins"),
            )
            ablation_result[str(group_name)] = group_runner.run()
            for bundle in group_bundles:
                stop_planning_bundle(bundle)

    return {
        "success": True,
        "reason": "benchmark_complete",
        "planner": planner_names if len(planner_names) != 1 else planner_names[0],
        "start_pos": start_pos.tolist(),
        "goal_pos": goal_pos.tolist(),
        "benchmark": bench_result,
        "ablation": ablation_result,
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.use_heuristic is not None:
        cfg.setdefault("experiment", {})["use_heuristic"] = bool(args.use_heuristic)
    if args.ablation_group is not None:
        cfg.setdefault("experiment", {})["ablation_group"] = str(args.ablation_group)
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = int(args.seed)
    if args.max_wall_time_s is not None:
        cfg.setdefault("experiment", {})["max_wall_time_s"] = float(args.max_wall_time_s)
    if args.save_video is not None:
        cfg.setdefault("experiment", {})["save_video"] = bool(args.save_video)

    exp_dir = setup_experiment_folder(cfg, args)
    logger = setup_logger(exp_dir, cfg)
    snapshot_code_and_config(exp_dir, cfg, args.config)

    exp_cfg = cfg.get("experiment", {})
    env_cfg = cfg.get("env", {})
    ctrl_cfg = cfg.get("control", {})
    vis_cfg = cfg.get("visualization", {})
    obstacle_cfg = cfg.get("obstacle", {})
    pointcloud_cfg = cfg.get("pointcloud", {})
    bench_cfg = cfg.get("benchmark", {})

    seed = int(exp_cfg.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    run_mode = str(exp_cfg.get("run_mode", "sim")).lower()
    planner_name = resolve_planner_name(exp_cfg, bench_cfg)
    if exp_cfg.get("planner") is not None:
        exp_cfg["use_heuristic"] = planner_name == "heuristic"

    logger.info("Experiment folder: %s", exp_dir)
    logger.info("Seed: %d", seed)
    logger.info("Run mode: %s", run_mode)
    logger.info("Planner: %s", planner_name)
    logger.info("Heuristic enabled: %s", bool(exp_cfg.get("use_heuristic", True)))
    logger.info("Ablation group: %s", str(exp_cfg.get("ablation_group", "full")))

    env_config_path = str(env_cfg.get("env_config_path", "src/config/env.yaml"))
    bounds_min, bounds_max = load_workspace_bounds(env_config_path)

    sim_env = PyBulletMotionEnv(env_cfg, logger)
    metrics_logger = MetricsLogger(exp_dir)

    planning_bundle = None
    video_log_id = None
    result = {
        "success": False,
        "reason": "unknown",
        "experiment_dir": str(exp_dir),
        "seed": seed,
        "run_mode": run_mode,
        "planner": planner_name,
        "use_heuristic": bool(exp_cfg.get("use_heuristic", True)),
        "ablation_group": str(exp_cfg.get("ablation_group", "full")),
        "error": None,
    }

    try:
        if run_mode == "benchmark":
            result.update(
                run_benchmark_mode(
                    cfg=cfg,
                    exp_cfg=exp_cfg,
                    env_cfg=env_cfg,
                    obstacle_cfg=obstacle_cfg,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    seed=seed,
                    logger=logger,
                    sim_env=sim_env,
                )
            )
            return

        client_id = sim_env.connect()
        proj_matrix = sim_env.compute_projection_matrix()

        predictor = create_predictor(
            config=cfg,
            use_heuristic=bool(exp_cfg.get("use_heuristic", True)),
            rng_seed=seed + 17,
            logger=logger,
        )

        obstacle_radius = float(obstacle_cfg.get("radius", 0.09))
        obstacle_center_cache = None

        def get_dynamic_obstacles(t_abs):
            _ = t_abs
            if obstacle_center_cache is None:
                return []
            return [SphereObstacle(center=np.asarray(obstacle_center_cache, dtype=np.float64), radius=obstacle_radius)]

        planning_bundle = create_planning_bundle(
            config=cfg,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            dynamic_obstacle_fn=get_dynamic_obstacles,
            predictor=predictor,
            rng=rng,
        )
        start_planning_bundle(planning_bundle)

        start_pos, goal_pos = resolve_start_goal(exp_cfg, sim_env)

        obstacle_center_fn = build_tube_motion(
            start_pos=start_pos,
            goal_pos=goal_pos,
            tube_radius=float(obstacle_cfg.get("tube_radius", 0.07)),
            endpoint_clearance=float(obstacle_cfg.get("endpoint_clearance", 0.18)),
        )

        obs_init = np.clip(obstacle_center_fn(0.0), bounds_min, bounds_max)
        obstacle_center_cache = np.asarray(obs_init, dtype=np.float64)
        obstacle_id = sim_env.create_sphere_obstacle(
            center=obs_init.tolist(),
            radius=obstacle_radius,
            rgba=obstacle_cfg.get("color_rgba", [1.0, 0.0, 0.0, 1.0]),
        )

        marker_radius = float(vis_cfg.get("marker_radius", 0.03))
        _ = sim_env.create_marker(start_pos.tolist(), radius=marker_radius, rgba=[0.1, 1.0, 0.1, 0.9])
        _ = sim_env.create_marker(goal_pos.tolist(), radius=marker_radius, rgba=[1.0, 0.1, 0.1, 0.9])

        sim_hz = float(ctrl_cfg.get("sim_hz", 240.0))
        sim_dt = 1.0 / max(sim_hz, 1.0)
        init_cfg = ctrl_cfg.get("init", {})
        reached_start = sim_env.move_to_target_pose(
            target_pos=start_pos,
            sim_dt=sim_dt,
            max_steps=int(init_cfg.get("max_steps", 1200)),
            reach_tol=float(init_cfg.get("reach_tol", 0.03)),
            max_step_rad=float(init_cfg.get("max_step_rad", 0.02)),
            position_gain=float(ctrl_cfg.get("joint_limits", {}).get("position_gain", 0.08)),
            velocity_gain=float(ctrl_cfg.get("joint_limits", {}).get("velocity_gain", 0.6)),
            fixed_ee_orn=None,
        )

        ee_after_init = sim_env.get_end_effector_pos()
        logger.info("Init EE-start error: %.4f m", np.linalg.norm(ee_after_init - start_pos))
        if not reached_start:
            logger.warning("Could not fully reach start pose before planning; continue with best-effort state.")

        safe_joint_posture = sim_env.get_joint_positions()

        video_log_id = start_video_recording(
            enabled=bool(exp_cfg.get("save_video", True)),
            output_path=str(exp_dir / str(exp_cfg.get("video_filename", "sim.mp4"))),
        )

        t = 0.0
        last_replan_t = -1e9
        last_periodic_replan_t = -1e9
        last_pointcloud_t = -1e9

        waypoint_reach_tol = float(ctrl_cfg.get("waypoint_reach_tol", 0.03))
        active_waypoint = None
        active_waypoint_last_progress_t = -1e9
        active_waypoint_best_dist = 1e9
        ik_fail_streak = 0

        shared_path = []
        shared_path_status = {}
        shared_path_version = 0
        pointcloud_version = 0
        cached_pc = None
        cached_pc_aabb = None

        wall_time_start = time.time()
        max_wall_time_s = float(exp_cfg.get("max_wall_time_s", 120.0))

        logger.info("Start control loop...")
        while sim_env.is_connected():
            if (time.time() - wall_time_start) > max_wall_time_s:
                logger.warning("Run timeout reached: %.2f s", max_wall_time_s)
                result["reason"] = "timeout"
                break

            t += sim_dt

            obs_pos = np.clip(obstacle_center_fn(t), bounds_min, bounds_max)
            obstacle_center_cache = np.asarray(obs_pos, dtype=np.float64)
            sim_env.update_obstacle_pose(obstacle_id, obs_pos.tolist())

            current_ee_pos = sim_env.get_end_effector_pos()

            can_replan_now = (t - last_replan_t) >= float(ctrl_cfg.get("replan_min_interval", 0.015))
            periodic_due = (t - last_periodic_replan_t) >= float(ctrl_cfg.get("periodic_online_replan_interval", 0.2))
            need_replan = (active_waypoint is None and len(shared_path) == 0) or periodic_due

            if can_replan_now and need_replan:
                last_replan_t = t
                if periodic_due:
                    last_periodic_replan_t = t

                if (t - last_pointcloud_t) >= float(pointcloud_cfg.get("refresh_interval", 0.15)):
                    depth_buffer, view_matrix = sim_env.capture_depth_sensor_frame(proj_matrix)
                    request_pointcloud_update(
                        bundle=planning_bundle,
                        depth_buffer=depth_buffer,
                        view_matrix=view_matrix,
                        proj_matrix=proj_matrix,
                        bounds_min=bounds_min,
                        bounds_max=bounds_max,
                        pointcloud_cfg=pointcloud_cfg,
                    )
                    last_pointcloud_t = t

                request_replan(
                    bundle=planning_bundle,
                    point_cloud=cached_pc,
                    static_obs_aabb=cached_pc_aabb,
                    current_pos=current_ee_pos,
                    goal_pos=goal_pos,
                    current_t=t,
                )

            new_pc_version, new_pc, new_pc_aabb = poll_pointcloud_update(planning_bundle, pointcloud_version)
            if new_pc is not None:
                pointcloud_version = new_pc_version
                cached_pc = new_pc
                cached_pc_aabb = new_pc_aabb

            new_version, new_path, status = poll_plan_update(planning_bundle, shared_path_version)
            if new_path is not None:
                shared_path_version = new_version
                shared_path = list(new_path)
                shared_path_status = {} if status is None else dict(status)

                metrics_logger.log_planning(sim_time_s=t, status=shared_path_status, path=shared_path)

                if bool(vis_cfg.get("show_path", True)) and shared_path_status.get("plan_ok", False):
                    sim_env.update_path_debug_lines(
                        shared_path,
                        color=vis_cfg.get("path_color", [0.0, 0.9, 1.0]),
                        line_width=float(vis_cfg.get("path_line_width", 2.5)),
                    )

                if bool(vis_cfg.get("show_heuristic_points", True)) and shared_path_status.get("heuristic_refreshed", False):
                    sim_env.update_heuristic_debug_points(
                        planning_bundle.planner.heuristic_points,
                        color=vis_cfg.get("heuristic_color", [0.2, 1.0, 0.2]),
                        point_size=float(vis_cfg.get("heuristic_point_size", 4.0)),
                    )

                logger.info(
                    "replan status: plan_ok=%s heuristic_ok=%s t=%.2f wall=%.2fms tree=%s",
                    shared_path_status.get("plan_ok", False),
                    shared_path_status.get("heuristic_ok", False),
                    t,
                    float(shared_path_status.get("planning_wall_time_ms", 0.0)),
                    shared_path_status.get("tree_size", 0),
                )

            if active_waypoint is None:
                while len(shared_path) > 0:
                    cand_waypoint, _ = shared_path.pop(0)
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
                elif (t - active_waypoint_last_progress_t) > float(ctrl_cfg.get("waypoint_stall_timeout", 0.7)):
                    logger.warning("Waypoint stalled. Drop current waypoint and force replan.")
                    active_waypoint = None
                    ik_fail_streak = 0
                    active_waypoint_best_dist = 1e9
                    shared_path = []
                    last_replan_t = -1e9
                    sim_env.command_joints(safe_joint_posture)
                else:
                    ee_to_wp = np.asarray(active_waypoint, dtype=np.float64) - current_ee_pos
                    ee_dist = float(np.linalg.norm(ee_to_wp))
                    ee_step = min(float(ctrl_cfg.get("ee_step", 0.012)), ee_dist)
                    if ee_dist > 1e-9:
                        ee_target = current_ee_pos + ee_to_wp / ee_dist * ee_step
                    else:
                        ee_target = np.asarray(active_waypoint, dtype=np.float64)

                    j_qs = sim_env.solve_ik_continuous(ee_target, None)
                    if j_qs is None:
                        ik_fail_streak += 1
                        sim_env.hold_current_posture(
                            position_gain=float(ctrl_cfg.get("joint_limits", {}).get("hold_position_gain", 0.06)),
                            velocity_gain=float(ctrl_cfg.get("joint_limits", {}).get("hold_velocity_gain", 0.5)),
                        )
                    else:
                        q_cmd = sim_env.compute_limited_joint_targets(
                            j_qs,
                            max_step_rad=float(ctrl_cfg.get("joint_limits", {}).get("max_step_rad", 0.012)),
                        )
                        if q_cmd is None:
                            ik_fail_streak += 1
                            sim_env.hold_current_posture(
                                position_gain=float(ctrl_cfg.get("joint_limits", {}).get("hold_position_gain", 0.06)),
                                velocity_gain=float(ctrl_cfg.get("joint_limits", {}).get("hold_velocity_gain", 0.5)),
                            )
                        else:
                            ik_fail_streak = 0
                            sim_env.command_joints(
                                q_cmd,
                                position_gain=float(ctrl_cfg.get("joint_limits", {}).get("position_gain", 0.08)),
                                velocity_gain=float(ctrl_cfg.get("joint_limits", {}).get("velocity_gain", 0.6)),
                            )

                    if ik_fail_streak >= int(ctrl_cfg.get("max_ik_fail_streak", 25)):
                        logger.warning("IK failed repeatedly. Fallback to safe posture and force replan.")
                        sim_env.command_joints(safe_joint_posture)
                        active_waypoint = None
                        ik_fail_streak = 0
                        active_waypoint_best_dist = 1e9
                        shared_path = []
                        last_replan_t = -1e9
            else:
                sim_env.hold_current_posture(
                    position_gain=float(ctrl_cfg.get("joint_limits", {}).get("hold_position_gain", 0.06)),
                    velocity_gain=float(ctrl_cfg.get("joint_limits", {}).get("hold_velocity_gain", 0.5)),
                )

            dist_to_goal = float(np.linalg.norm(current_ee_pos - goal_pos))
            if dist_to_goal < 0.05:
                logger.info("Goal reached.")
                result["success"] = True
                result["reason"] = "goal_reached"
                break

            if not sim_env.safe_step_sim():
                logger.error("Physics server disconnected during control loop.")
                result["reason"] = "physics_disconnected"
                break
            time.sleep(sim_dt)

        stats = planning_bundle.planning_manager.get_stats_snapshot() if planning_bundle is not None else {}
        result["planning_stats"] = stats
        result["metrics_rows"] = metrics_logger.rows
        result["elapsed_wall_time_s"] = float(time.time() - wall_time_start)
        if result["reason"] == "unknown":
            result["reason"] = "finished"

    except Exception as exc:
        logger.exception("Run failed with exception.")
        result["success"] = False
        result["reason"] = "exception"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        stop_video_recording(video_log_id)
        if planning_bundle is not None:
            stop_planning_bundle(planning_bundle)
        sim_env.disconnect()
        save_result_json(exp_dir, result)


if __name__ == "__main__":
    main()
