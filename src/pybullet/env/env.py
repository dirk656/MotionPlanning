import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data


class PyBulletMotionEnv:
	"""PyBullet environment wrapper for robot control, sensing, and debug rendering."""

	def __init__(self, config: Dict, logger):
		self.config = config
		self.logger = logger
		self.client_id = -1
		self.robot_id = None
		self.end_effector_idx = None
		self.joint_indices: List[int] = []
		self.path_debug_line_ids: List[int] = []
		self.heuristic_debug_item_id: Optional[int] = None

	def connect(self) -> int:
		mode = str(self.config.get("pybullet_mode", "GUI")).upper()
		if mode == "DIRECT":
			self.client_id = p.connect(p.DIRECT)
		else:
			self.client_id = p.connect(p.GUI)
			if self.client_id < 0:
				self.client_id = p.connect(p.DIRECT)

		if self.client_id < 0:
			raise RuntimeError("Failed to connect to PyBullet")

		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		gravity = self.config.get("gravity", [0.0, 0.0, -9.81])
		p.setGravity(float(gravity[0]), float(gravity[1]), float(gravity[2]))

		urdf_cfg = self.config.get("urdf", {})
		p.loadURDF(str(urdf_cfg.get("plane", "plane.urdf")))
		self.robot_id = p.loadURDF(
			str(urdf_cfg.get("robot", "franka_panda/panda.urdf")),
			[0, 0, 0],
			useFixedBase=True,
		)

		robot_cfg = self.config.get("robot", {})
		self.end_effector_idx = int(robot_cfg.get("end_effector_idx", 11))
		self.joint_indices = [int(v) for v in robot_cfg.get("controlled_joint_indices", list(range(7)))]
		self.logger.info("PyBullet connected: client_id=%s, robot_id=%s", self.client_id, self.robot_id)
		return self.client_id

	def disconnect(self):
		if self.client_id >= 0 and p.isConnected(self.client_id):
			p.disconnect(self.client_id)
		self.client_id = -1

	def is_connected(self) -> bool:
		return self.client_id >= 0 and p.isConnected(self.client_id)

	def compute_projection_matrix(self):
		cam = self.config.get("camera", {})
		width = int(cam.get("width", 320))
		height = int(cam.get("height", 240))
		return p.computeProjectionMatrixFOV(
			fov=float(cam.get("fov", 60.0)),
			aspect=width / max(height, 1),
			nearVal=float(cam.get("near", 0.05)),
			farVal=float(cam.get("far", 3.0)),
		)

	def _compute_view_matrix(self):
		cam = self.config.get("camera", {})
		return p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=[float(v) for v in cam.get("target", [0.25, 0.0, 0.25])],
			distance=float(cam.get("distance", 1.3)),
			yaw=float(cam.get("yaw", 45.0)),
			pitch=float(cam.get("pitch", -35.0)),
			roll=float(cam.get("roll", 0.0)),
			upAxisIndex=int(cam.get("up_axis", 2)),
		)

	def capture_depth_sensor_frame(self, proj_matrix):
		cam = self.config.get("camera", {})
		view_matrix = self._compute_view_matrix()
		_, _, _, depth_buffer, _ = p.getCameraImage(
			width=int(cam.get("width", 320)),
			height=int(cam.get("height", 240)),
			viewMatrix=view_matrix,
			projectionMatrix=proj_matrix,
			renderer=p.ER_BULLET_HARDWARE_OPENGL,
		)
		return np.asarray(depth_buffer, dtype=np.float64), np.asarray(view_matrix, dtype=np.float64)

	def get_end_effector_pos(self) -> np.ndarray:
		state = p.getLinkState(self.robot_id, self.end_effector_idx)
		return np.asarray(state[4], dtype=np.float64)

	def solve_ik_continuous(self, target_pos: np.ndarray, target_orn=None):
		if not self.is_connected():
			return None
		try:
			rest_q = [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
			kwargs = {
				"bodyUniqueId": self.robot_id,
				"endEffectorLinkIndex": self.end_effector_idx,
				"targetPosition": np.asarray(target_pos, dtype=np.float64).tolist(),
				"restPoses": rest_q,
			}
			if target_orn is not None:
				kwargs["targetOrientation"] = target_orn
			ik = p.calculateInverseKinematics(**kwargs)
			return list(ik)[: len(self.joint_indices)]
		except Exception:
			return None

	def compute_limited_joint_targets(self, raw_targets, max_step_rad=0.02):
		if not self.is_connected():
			return None
		curr_q = np.array([p.getJointState(self.robot_id, j)[0] for j in self.joint_indices], dtype=np.float64)
		raw_q = np.asarray(raw_targets, dtype=np.float64)
		delta = np.clip(raw_q - curr_q, -float(max_step_rad), float(max_step_rad))
		return (curr_q + delta).tolist()

	def command_joints(self, q_cmd, position_gain=0.08, velocity_gain=0.6):
		if q_cmd is None:
			return
		p.setJointMotorControlArray(
			self.robot_id,
			self.joint_indices,
			p.POSITION_CONTROL,
			targetPositions=list(q_cmd),
			positionGains=[float(position_gain)] * len(self.joint_indices),
			velocityGains=[float(velocity_gain)] * len(self.joint_indices),
		)

	def hold_current_posture(self, position_gain=0.06, velocity_gain=0.5):
		q_hold = [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
		self.command_joints(q_hold, position_gain=position_gain, velocity_gain=velocity_gain)

	def reset_joint_positions(self, q):
		for i, j in enumerate(self.joint_indices):
			p.resetJointState(self.robot_id, j, float(q[i]))

	def get_joint_positions(self):
		return [p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]

	def safe_step_sim(self) -> bool:
		if not self.is_connected():
			return False
		try:
			p.stepSimulation()
			return True
		except Exception:
			return False

	def create_marker(self, position, radius, rgba):
		visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
		collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=float(radius))
		return p.createMultiBody(
			baseMass=0,
			baseCollisionShapeIndex=collision_id,
			baseVisualShapeIndex=visual_id,
			basePosition=list(position),
		)

	def create_sphere_obstacle(self, center, radius, rgba):
		visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
		collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=float(radius))
		obstacle_id = p.createMultiBody(
			baseMass=0,
			baseCollisionShapeIndex=collision_shape_id,
			baseVisualShapeIndex=visual_shape_id,
			basePosition=list(center),
		)
		return obstacle_id

	def update_obstacle_pose(self, obstacle_id, pos):
		if self.is_connected():
			p.resetBasePositionAndOrientation(obstacle_id, list(pos), [0, 0, 0, 1])

	def update_path_debug_lines(self, path, color=None, line_width=2.5):
		if color is None:
			color = [0.0, 0.9, 1.0]

		for line_id in self.path_debug_line_ids:
			try:
				p.removeUserDebugItem(line_id)
			except Exception:
				pass
		self.path_debug_line_ids = []

		if path is None or len(path) < 2:
			return

		for idx in range(len(path) - 1):
			p0 = np.asarray(path[idx][0], dtype=np.float64)
			p1 = np.asarray(path[idx + 1][0], dtype=np.float64)
			try:
				line_id = p.addUserDebugLine(
					p0.tolist(),
					p1.tolist(),
					lineColorRGB=list(color),
					lineWidth=float(line_width),
					lifeTime=0,
				)
				self.path_debug_line_ids.append(line_id)
			except Exception:
				break

	def update_heuristic_debug_points(self, points, color=None, point_size=4.0):
		if color is None:
			color = [0.2, 1.0, 0.2]

		if self.heuristic_debug_item_id is not None:
			try:
				p.removeUserDebugItem(self.heuristic_debug_item_id)
			except Exception:
				pass
			self.heuristic_debug_item_id = None

		if points is None:
			return
		points = np.asarray(points, dtype=np.float64)
		if points.ndim != 2 or points.shape[0] == 0:
			return

		try:
			colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (points.shape[0], 1))
			self.heuristic_debug_item_id = p.addUserDebugPoints(
				pointPositions=points.tolist(),
				pointColorsRGB=colors.tolist(),
				pointSize=float(point_size),
				lifeTime=0,
			)
		except Exception:
			self.heuristic_debug_item_id = None

	def move_to_target_pose(
		self,
		target_pos,
		sim_dt,
		max_steps=1200,
		reach_tol=0.03,
		max_step_rad=0.02,
		position_gain=0.08,
		velocity_gain=0.6,
		fixed_ee_orn=None,
	):
		reached = False
		for _ in range(int(max_steps)):
			if not self.is_connected():
				break
			ee_now = self.get_end_effector_pos()
			if np.linalg.norm(ee_now - np.asarray(target_pos, dtype=np.float64)) <= float(reach_tol):
				reached = True
				break
			q_raw = self.solve_ik_continuous(target_pos=np.asarray(target_pos, dtype=np.float64), target_orn=fixed_ee_orn)
			if q_raw is None:
				continue
			q_cmd = self.compute_limited_joint_targets(q_raw, max_step_rad=max_step_rad)
			self.command_joints(q_cmd, position_gain=position_gain, velocity_gain=velocity_gain)
			if not self.safe_step_sim():
				break
			time.sleep(float(sim_dt))
		return reached


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

	endpoint_margin = float(np.clip(float(endpoint_clearance) / max(seg_len, 1e-9), 0.0, 0.45))
	alpha_min = endpoint_margin
	alpha_max = 1.0 - endpoint_margin

	def obstacle_pos(t_abs):
		line_phase = 0.5 * (1.0 + np.sin(0.9 * float(t_abs)))
		line_phase = float(np.clip(line_phase, alpha_min, alpha_max))
		swirl_phase = 2.3 * float(t_abs)
		base = start_pos + line_phase * seg
		offset = float(tube_radius) * (
			np.cos(swirl_phase) * n1 + np.sin(swirl_phase) * n2
		)
		return base + offset

	return obstacle_pos


def sample_reachable_start_goal(
	ik_solver: Callable,
	get_ee_position: Callable,
	reset_joint_positions: Callable,
	get_joint_positions: Callable,
	generate_pos_fn: Callable,
	max_tries=60,
):
	for _ in range(int(max_tries)):
		start_pos, goal_pos = generate_pos_fn(
			max_radius=0.62,
			min_start_goal_distance=0.55,
			min_robot_distance_xy=0.46,
			min_robot_distance_3d=0.56,
			min_line_robot_distance_3d=0.44,
			z_range=(0.28, 0.88),
			max_attempts=800,
		)

		q_s = ik_solver(start_pos, None)
		q_g = ik_solver(goal_pos, None)
		if q_s is None or q_g is None:
			continue

		old_states = get_joint_positions()
		reset_joint_positions(q_s)
		ee_s = np.asarray(get_ee_position(), dtype=np.float64)
		err_s = float(np.linalg.norm(ee_s - start_pos))

		reset_joint_positions(q_g)
		ee_g = np.asarray(get_ee_position(), dtype=np.float64)
		err_g = float(np.linalg.norm(ee_g - goal_pos))

		reset_joint_positions(old_states)
		if err_s <= 0.05 and err_g <= 0.05:
			return np.asarray(start_pos, dtype=np.float64), np.asarray(goal_pos, dtype=np.float64)

	raise RuntimeError("Failed to sample IK-reachable start/goal pair")


def start_video_recording(enabled: bool, output_path: str):
	if not enabled:
		return None
	try:
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		return p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, output_path)
	except Exception:
		return None


def stop_video_recording(log_id):
	if log_id is None:
		return
	try:
		p.stopStateLogging(log_id)
	except Exception:
		pass
