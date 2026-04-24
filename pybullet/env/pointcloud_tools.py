#generate start pos and goal pos for pybullet environment
import numpy as np
import yaml 


def load_workspace_bounds(env_config_path):
    with open(env_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bounds = config.get("environment", {}).get("workspace_bounds")
    if not isinstance(bounds, list) or len(bounds) != 6:
        raise ValueError("environment.workspace_bounds must be a list of 6 numbers")

    bounds = np.asarray(bounds, dtype=np.float64)
    bounds_min = bounds[:3]
    bounds_max = bounds[3:]
    if np.any(bounds_min >= bounds_max):
        raise ValueError("workspace_bounds must satisfy min < max for x,y,z")
    return bounds_min, bounds_max




def filter_points_by_bounds(points, bounds_min, bounds_max):
    mask = np.all((points >= bounds_min) & (points <= bounds_max), axis=1)
    return points[mask]



def depth_buffer_to_points_world(depth_buffer, view_matrix, proj_matrix):
    """Convert PyBullet depth buffer to world-frame point cloud."""
    h, w = depth_buffer.shape

    xs = np.linspace(-1.0, 1.0, w)
    ys = np.linspace(1.0, -1.0, h)
    x_grid, y_grid = np.meshgrid(xs, ys)

    z_clip = depth_buffer * 2.0 - 1.0
    ones = np.ones_like(z_clip)
    clip_coords = np.stack([x_grid, y_grid, z_clip, ones], axis=-1).reshape(-1, 4)

    view = np.array(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
    proj = np.array(proj_matrix, dtype=np.float64).reshape(4, 4, order="F")
    inv_vp = np.linalg.inv(proj @ view)

    world_h = clip_coords @ inv_vp.T
    world = world_h[:, :3] / np.clip(world_h[:, 3:4], 1e-8, None)

    valid = np.isfinite(world).all(axis=1)
    valid &= depth_buffer.reshape(-1) < 0.999
    return world[valid]


    
