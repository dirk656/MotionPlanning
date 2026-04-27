#generate start pos and goal pos for pybullet environment
import numpy as np
import threading
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


def voxelize_points(points, voxel_size=0.03, max_voxels=1200):
    """Voxelize point cloud and return voxel centroids and voxel AABBs."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")
    if points.shape[0] == 0:
        empty_xyz = np.empty((0, 3), dtype=np.float64)
        empty_aabb = np.empty((0, 6), dtype=np.float64)
        return empty_xyz, empty_aabb

    voxel_size = float(max(voxel_size, 1e-6))
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)

    unique_idx, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    num_voxels = unique_idx.shape[0]

    sums = np.zeros((num_voxels, 3), dtype=np.float64)
    counts = np.zeros(num_voxels, dtype=np.int64)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    centroids = sums / counts[:, None]

    if max_voxels is not None:
        max_voxels = int(max(max_voxels, 1))
        if num_voxels > max_voxels:
            # Keep densest voxels first to preserve obstacle surfaces.
            keep_idx = np.argpartition(counts, -max_voxels)[-max_voxels:]
            centroids = centroids[keep_idx]
            unique_idx = unique_idx[keep_idx]

    mins = unique_idx.astype(np.float64) * voxel_size
    dims = np.full((mins.shape[0], 3), voxel_size, dtype=np.float64)
    voxel_aabb = np.concatenate([mins, dims], axis=1)
    return centroids, voxel_aabb


class AtomicPointCloudSnapshot:
    """Thread-safe latest point cloud buffer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._version = 0
        self._points = None
        self._aabb = None

    def publish(self, points, aabb):
        with self._lock:
            self._points = None if points is None else np.asarray(points, dtype=np.float64).copy()
            self._aabb = None if aabb is None else np.asarray(aabb, dtype=np.float64).copy()
            self._version += 1
            return self._version

    def get_if_newer(self, last_version):
        with self._lock:
            if self._version <= int(last_version):
                return int(last_version), None, None
            pts = None if self._points is None else self._points.copy()
            aabb = None if self._aabb is None else self._aabb.copy()
            return self._version, pts, aabb


class AsyncPointCloudNode:
    """Background node: depth-buffer to voxel cloud + voxel AABB."""

    def __init__(self):
        self.result_store = AtomicPointCloudSnapshot()
        self._request_lock = threading.Lock()
        self._latest_request = None
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="AsyncPointCloudNode", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._request_event.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(
        self,
        depth_buffer,
        view_matrix,
        proj_matrix,
        bounds_min,
        bounds_max,
        voxel_size=0.03,
        max_voxels=1200,
    ):
        req = {
            "depth_buffer": np.asarray(depth_buffer, dtype=np.float64).copy(),
            "view_matrix": np.asarray(view_matrix, dtype=np.float64).copy(),
            "proj_matrix": np.asarray(proj_matrix, dtype=np.float64).copy(),
            "bounds_min": np.asarray(bounds_min, dtype=np.float64).copy(),
            "bounds_max": np.asarray(bounds_max, dtype=np.float64).copy(),
            "voxel_size": float(voxel_size),
            "max_voxels": int(max_voxels),
        }
        with self._request_lock:
            self._latest_request = req
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

            points = depth_buffer_to_points_world(
                req["depth_buffer"],
                req["view_matrix"],
                req["proj_matrix"],
            )
            points = filter_points_by_bounds(points, req["bounds_min"], req["bounds_max"])
            voxel_points, voxel_aabb = voxelize_points(
                points,
                voxel_size=req["voxel_size"],
                max_voxels=req["max_voxels"],
            )
            self.result_store.publish(voxel_points, voxel_aabb)


    
