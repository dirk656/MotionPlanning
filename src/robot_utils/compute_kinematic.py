import xml.etree.ElementTree as ET
import numpy as np
from robot_utils.rotation_matrix import *
import pinocchio as pin
from robot_utils.urdf_to_geometry import parse_urdf



def build_pin_model(urdf_path, package_dirs=None, root_joint=None):
	"""
	从 URDF 构建 Pinocchio 模型。
	"""

	if root_joint is None:
		model = pin.buildModelFromUrdf(urdf_path)
	else:
		model = pin.buildModelFromUrdf(urdf_path, root_joint)
	data = model.createData()
	return model, data


def _frame_id_from_name(model, frame_name):
	if frame_name not in [f.name for f in model.frames]:
		raise ValueError(f"Frame '{frame_name}' not found in model frames")
	return model.getFrameId(frame_name)


def make_target_se3(target_xyz, target_rpy):
	"""
	使用 xyz + rpy 构造目标位姿。
	"""
	
	R = rpy_to_rotation(np.asarray(target_rpy, dtype=float))
	t = np.asarray(target_xyz, dtype=float).reshape(3)
	return pin.SE3(R, t)


def compute_fk(model, data, q, ee_frame_name):
	"""
	计算末端执行器的 FK，返回 SE3 和 numpy 结果。
	"""
	
	q = np.asarray(q, dtype=float).reshape(model.nq)
	frame_id = _frame_id_from_name(model, ee_frame_name)

	pin.forwardKinematics(model, data, q)
	pin.updateFramePlacements(model, data)

	pose = data.oMf[frame_id]
	return {
		"se3": pose,
		"xyz": pose.translation.copy(),
		"rotation": pose.rotation.copy(),
	}


def compute_ik(
	model,
	data,
	ee_frame_name,
	target_xyz,
	target_rpy,
	q0=None,
	max_iter=200,
	tol=1e-4,
	damping=1e-6,
	step_size=1.0,
	with_orientation=True,
):
	"""
	使用阻尼最小二乘迭代求解 IK。
	"""
	
	frame_id = _frame_id_from_name(model, ee_frame_name)
	target = make_target_se3(target_xyz, target_rpy)

	if q0 is None:
		q = pin.neutral(model)
	else:
		q = np.asarray(q0, dtype=float).reshape(model.nq)

	for i in range(max_iter):
		pin.forwardKinematics(model, data, q)
		pin.updateFramePlacements(model, data)

		current = data.oMf[frame_id]
		err_se3 = current.inverse() * target
		err6 = pin.log(err_se3).vector

		if with_orientation:
			err = err6
			J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)
		else:
			err = target.translation - current.translation
			J6 = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
			J = J6[:3, :]

		if np.linalg.norm(err) < tol:
			fk = compute_fk(model, data, q, ee_frame_name)
			return {
				"success": True,
				"q": q.copy(),
				"iterations": i,
				"error_norm": float(np.linalg.norm(err)),
				"fk": fk,
			}

		JJt = J @ J.T
		delta = J.T @ np.linalg.solve(JJt + damping * np.eye(JJt.shape[0]), err)
		q = pin.integrate(model, q, step_size * delta)

	fk = compute_fk(model, data, q, ee_frame_name)
	final_err = np.linalg.norm(err)
	return {
		"success": False,
		"q": q.copy(),
		"iterations": max_iter,
		"error_norm": float(final_err),
		"fk": fk,
	}

