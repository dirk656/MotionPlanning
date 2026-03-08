import numpy as np



def rotation_matrix_axis_angle(axis, angle):
    """绕任意轴的旋转矩阵（Rodrigues 公式）"""
    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rpy_to_rotation(rpy):
    """RPY (roll, pitch, yaw) → 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    r, p, y = rpy
    Rx = rotation_matrix_axis_angle([1, 0, 0], r)
    Ry = rotation_matrix_axis_angle([0, 1, 0], p)
    Rz = rotation_matrix_axis_angle([0, 0, 1], y)
    return Rz @ Ry @ Rx


def make_transform(xyz, rpy):
    """构造 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = rpy_to_rotation(rpy)
    T[:3, 3] = xyz
    return T


def parse_vec(s, default=(0, 0, 0)):
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.split()], dtype=float)
