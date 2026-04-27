import numpy as np
import yaml
from typing import Optional



def generate_pos(
    max_radius: float,
    min_start_goal_distance: float = 0.4,
    min_robot_distance_xy: float = 0.35,
    min_robot_distance_3d: float = 0.45,
    min_line_robot_distance_3d: Optional[float] = None,
    z_range: tuple[float, float] = (0.2, 0.8),
    max_attempts: int = 1000,
    robot_base: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample start/goal points in workspace with distance constraints."""

    def sample_one() -> np.ndarray:
        # Uniform sample in disk (XY) + constrained Z.
        r = np.sqrt(np.random.uniform(0.0, max_radius * max_radius))
        theta = np.random.uniform(-np.pi, np.pi)
        z = np.random.uniform(z_range[0], z_range[1])
        return np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=np.float64)

    def point_to_segment_distance(point, seg_start, seg_end) -> float:
        seg = seg_end - seg_start
        seg_norm_sq = float(np.dot(seg, seg))
        if seg_norm_sq < 1e-12:
            return float(np.linalg.norm(point - seg_start))
        u = float(np.dot(point - seg_start, seg) / seg_norm_sq)
        u = float(np.clip(u, 0.0, 1.0))
        nearest = seg_start + u * seg
        return float(np.linalg.norm(point - nearest))

    robot_base = np.asarray(robot_base, dtype=np.float64)
    for _ in range(max_attempts):
        start_pos = sample_one()
        goal_pos = sample_one()

        if np.linalg.norm(start_pos[:2] - robot_base[:2]) < min_robot_distance_xy:
            continue
        if np.linalg.norm(goal_pos[:2] - robot_base[:2]) < min_robot_distance_xy:
            continue
        if np.linalg.norm(start_pos - robot_base) < min_robot_distance_3d:
            continue
        if np.linalg.norm(goal_pos - robot_base) < min_robot_distance_3d:
            continue
        if np.linalg.norm(start_pos - goal_pos) < min_start_goal_distance:
            continue
        if (
            min_line_robot_distance_3d is not None
            and point_to_segment_distance(robot_base, start_pos, goal_pos) < min_line_robot_distance_3d
        ):
            continue

        return start_pos, goal_pos

    raise RuntimeError(
        "Failed to sample valid start/goal positions. Consider reducing constraints or increasing max_radius."
    )



def in_bound(point , bounds_min , bounds_max) -> bool:
    return np.all(point >= bounds_min) and np.all(point <= bounds_max)

