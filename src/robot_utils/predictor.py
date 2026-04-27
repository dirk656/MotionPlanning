import os

import numpy as np
import yaml

from wrapper_3d.pointnet_pointnet2.pointnet2_wrapper_connect_bfs import PNGWrapper


class HeuristicRegionPredictor:
    """Predict heuristic region points that likely connect start and goal."""

    def __init__(
        self,
        model_root="results/model_training/pointnet2_3d/checkpoints/best_pointnet2_3d",
        env_config_path=None,
        device="cuda",
        max_predict_points=4096,
        rng_seed=42,
    ):
        if env_config_path is None:
            env_config_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "env.yaml"
            )

        with open(env_config_path, "r", encoding="utf-8") as f:
            self.env_dict = yaml.safe_load(f)
        self.wrapper_env_dict = self._build_wrapper_env_dict(self.env_dict)
        self.max_predict_points = int(max_predict_points)
        self.rng = np.random.default_rng(rng_seed)

        self.wrapper = PNGWrapper(root_dir=model_root, device=device)

    @staticmethod
    def _build_wrapper_env_dict(raw_env_dict):
        if "env_dims" in raw_env_dict:
            return raw_env_dict

        env_section = raw_env_dict.get("environment", {})
        workspace_bounds = env_section.get("workspace_bounds", None)
        if workspace_bounds is None or len(workspace_bounds) != 6:
            raise KeyError(
                "env.yaml must include either 'env_dims' or 'environment.workspace_bounds' with 6 values"
            )

        xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in workspace_bounds]
        env_width = max(1e-6, xmax - xmin)
        env_height = max(1e-6, ymax - ymin)
        env_depth = max(1e-6, zmax - zmin)

        wrapper_env_dict = dict(raw_env_dict)
        wrapper_env_dict["env_dims"] = (env_height, env_width, env_depth)
        return wrapper_env_dict

    def predict(
        self,
        pc,
        x_start,
        x_goal,
        neighbor_radius=0.5,
        max_trial_attempts=5,
        visualize=False,
    ):
        """
        Return:
            success: bool
            heuristic_points: np.ndarray, shape [M, 3]
            path_pred_mask: np.ndarray, shape [N]
            num_runs: int
        """
        pc = np.asarray(pc, dtype=np.float32)
        x_start = np.asarray(x_start, dtype=np.float32)
        x_goal = np.asarray(x_goal, dtype=np.float32)

        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError("pc must be an array with shape [N, 3]")

        n_points = pc.shape[0]
        sampled_idx = None
        pc_for_model = pc

        # PointNet++ inference can be expensive on large raw clouds, especially on CPU.
        if n_points > self.max_predict_points:
            sampled_idx = self.rng.choice(n_points, size=self.max_predict_points, replace=False)
            pc_for_model = pc[sampled_idx]

        success, num_runs, path_pred_mask = self.wrapper.generate_connected_path_points(
            pc=pc_for_model,
            x_start=x_start,
            x_goal=x_goal,
            env_dict=self.wrapper_env_dict,
            neighbor_radius=neighbor_radius,
            max_trial_attempts=max_trial_attempts,
            visualize=visualize,
        )

        path_pred_mask = np.asarray(path_pred_mask, dtype=np.float32)

        if sampled_idx is not None:
            full_mask = np.zeros((n_points,), dtype=np.float32)
            full_mask[sampled_idx] = path_pred_mask
            path_pred_mask = full_mask

        if success:
            heuristic_points = pc[np.asarray(path_pred_mask) == 1]
        else:
            heuristic_points = np.empty((0, 3), dtype=np.float32)

        return success, heuristic_points, path_pred_mask, num_runs
   