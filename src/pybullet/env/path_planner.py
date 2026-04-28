import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

import numpy as np

from src.pybullet.env.pointcloud_tools import AsyncPointCloudNode
from src.robot_utils.predictor import HeuristicRegionPredictor
from src.robot_utils.time_based_rrt import (
	AsyncPlanningNode,
	HeuristicTimeRRTPlanner,
	OnlineHeuristicPlanningManager,
)


class PredictorInterface(Protocol):
	def predict(self, pc, x_start, x_goal, neighbor_radius=0.5, max_trial_attempts=5, visualize=False):
		...


class PlannerInterface(Protocol):
	def set_heuristic_points(self, heuristic_points):
		...


@dataclass
class PlanningBundle:
	planner: HeuristicTimeRRTPlanner
	planning_manager: OnlineHeuristicPlanningManager
	planning_node: AsyncPlanningNode
	pointcloud_node: AsyncPointCloudNode


def create_predictor(config: Dict, use_heuristic: bool, rng_seed: int, logger) -> Optional[PredictorInterface]:
	if not use_heuristic:
		logger.info("Heuristic predictor disabled: baseline RRT mode.")
		return None

	model_cfg = config.get("model", {})
	try:
		predictor = HeuristicRegionPredictor(
			model_root=str(model_cfg.get("predictor_model_root", "results/model_training/pointnet2_3d/checkpoints/best_pointnet2_3d")),
			env_config_path=str(model_cfg.get("env_config_path", "src/config/env.yaml")),
			device=str(model_cfg.get("device", "cpu")),
			max_predict_points=int(model_cfg.get("max_predict_points", 4096)),
			rng_seed=int(rng_seed),
		)
		logger.info("Heuristic predictor loaded successfully.")
		return predictor
	except Exception as exc:
		logger.exception("Predictor load failed. Falling back to baseline RRT. error=%s", exc)
		return None


def create_planning_bundle(
	config: Dict,
	bounds_min,
	bounds_max,
	dynamic_obstacle_fn: Callable,
	predictor,
	rng,
) -> PlanningBundle:
	planning_cfg = config.get("planning", {})
	planner_cfg = planning_cfg.get("planner", {})
	manager_cfg = planning_cfg.get("manager", {})
	async_cfg = planning_cfg.get("async", {})

	planner = HeuristicTimeRRTPlanner(
		bounds_min=bounds_min,
		bounds_max=bounds_max,
		speed=float(planner_cfg.get("speed", 0.5)),
		step_len=float(planner_cfg.get("step_len", 0.035)),
		goal_tolerance=float(planner_cfg.get("goal_tolerance", 0.05)),
		goal_bias=float(planner_cfg.get("goal_bias", 0.25)),
		heuristic_bias=float(planner_cfg.get("heuristic_bias", 0.55)),
		max_iters=int(planner_cfg.get("max_iters", 260)),
		time_horizon=float(planner_cfg.get("time_horizon", 10.0)),
		collision_margin=float(planner_cfg.get("collision_margin", 0.05)),
		max_subtrees=int(planner_cfg.get("max_subtrees", 4)),
		subtree_spawn_prob=float(planner_cfg.get("subtree_spawn_prob", 0.12)),
		subtree_merge_distance=float(planner_cfg.get("subtree_merge_distance", 0.12)),
		risk_lookahead_s=float(planner_cfg.get("risk_lookahead_s", 0.35)),
		risk_samples=int(planner_cfg.get("risk_samples", 3)),
		risk_clearance_scale=float(planner_cfg.get("risk_clearance_scale", 1.0)),
		singularity_threshold=planner_cfg.get("singularity_threshold", None),
		singularity_penalty_gain=float(planner_cfg.get("singularity_penalty_gain", 2.0)),
		failed_region_radius=float(planner_cfg.get("failed_region_radius", 0.10)),
		failed_region_spawn_threshold=int(planner_cfg.get("failed_region_spawn_threshold", 3)),
	)

	planning_manager = OnlineHeuristicPlanningManager(
		planner=planner,
		predictor=predictor,
		dynamic_obstacle_fn=dynamic_obstacle_fn,
		arm_collision_checker=None,
		neighbor_radius=float(manager_cfg.get("neighbor_radius", 0.1)),
		max_trial_attempts=int(manager_cfg.get("max_trial_attempts", 3)),
		incremental_iter_budget=int(manager_cfg.get("incremental_iter_budget", 80)),
		tree_resume_pos_tol=float(manager_cfg.get("tree_resume_pos_tol", 0.08)),
		tree_resume_time_window=float(manager_cfg.get("tree_resume_time_window", 0.35)),
		heuristic_update_interval=float(manager_cfg.get("heuristic_update_interval", 0.15)),
		bias_decay=float(manager_cfg.get("bias_decay", 0.85)),
		bias_recover=float(manager_cfg.get("bias_recover", 0.05)),
		bias_decay_fail_threshold=int(manager_cfg.get("bias_decay_fail_threshold", 3)),
		min_heuristic_bias=float(manager_cfg.get("min_heuristic_bias", 0.05)),
		path_smoothing_trials=int(manager_cfg.get("path_smoothing_trials", 20)),
		path_smoothing_enabled=bool(manager_cfg.get("path_smoothing_enabled", True)),
		arm_singularity_checker=None,
		stuck_time_s=float(manager_cfg.get("stuck_time_s", 10.0)),
		stuck_goal_improve_eps=float(manager_cfg.get("stuck_goal_improve_eps", 0.01)),
		rng=rng,
	)

	planning_node = AsyncPlanningNode(
		planning_manager=planning_manager,
		iter_budget=int(async_cfg.get("iter_budget", 80)),
		visualize=bool(async_cfg.get("visualize", False)),
	)
	pointcloud_node = AsyncPointCloudNode()

	return PlanningBundle(
		planner=planner,
		planning_manager=planning_manager,
		planning_node=planning_node,
		pointcloud_node=pointcloud_node,
	)


def start_planning_bundle(bundle: PlanningBundle):
	bundle.planning_node.start()
	bundle.pointcloud_node.start()


def stop_planning_bundle(bundle: PlanningBundle):
	bundle.planning_node.stop()
	bundle.planning_node.join(timeout=1.0)
	bundle.pointcloud_node.stop()
	bundle.pointcloud_node.join(timeout=1.0)


def request_pointcloud_update(bundle: PlanningBundle, depth_buffer, view_matrix, proj_matrix, bounds_min, bounds_max, pointcloud_cfg: Dict):
	bundle.pointcloud_node.submit(
		depth_buffer=depth_buffer,
		view_matrix=view_matrix,
		proj_matrix=np.asarray(proj_matrix, dtype=np.float64),
		bounds_min=bounds_min,
		bounds_max=bounds_max,
		voxel_size=float(pointcloud_cfg.get("voxel_size", 0.03)),
		max_voxels=int(pointcloud_cfg.get("max_voxels", 1200)),
	)


def request_replan(bundle: PlanningBundle, point_cloud, static_obs_aabb, current_pos, goal_pos, current_t):
	bundle.planning_node.submit(
		point_cloud=point_cloud,
		static_obs_aabb=static_obs_aabb,
		current_pos=current_pos,
		goal_pos=goal_pos,
		current_t=current_t,
	)


def poll_pointcloud_update(bundle: PlanningBundle, last_version: int):
	return bundle.pointcloud_node.result_store.get_if_newer(last_version)


def poll_plan_update(bundle: PlanningBundle, last_version: int):
	return bundle.planning_node.result_store.get_if_newer(last_version)

