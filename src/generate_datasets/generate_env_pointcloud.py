# pyright: reportMissingImports=false
import argparse
import glob
import json
import os
from os.path import join

import numpy as np
import yaml

from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points
from datasets_3d.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d_v1
from planning_utils.env import Env


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ENV_CFG_PATH = os.path.join(_PROJECT_ROOT, "src", "config", "env.yaml")


def _resolve_from_project(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(_PROJECT_ROOT, path_str)


def load_env_config(cfg_path):
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def _scene_workspace(scene, env_cfg):
    if "workspace_bounds" in scene:
        ws = np.asarray(scene["workspace_bounds"], dtype=float)
    else:
        ws = np.asarray(env_cfg["environment"]["workspace_bounds"], dtype=float)
    return ws


def _convert_obstacles_to_env_local(scene, ws):
    """
    将场景障碍物转换为 Env 所需格式。
    Env 的 box 输入格式: [x_min, y_min, z_min, sx, sy, sz]
    Env 的 ball 输入格式: [cx, cy, cz, r]
    """
    origin = ws[:3]
    box_obs = []
    ball_obs = []

    for obs in scene["obstacles"]:
        otype = obs["type"]
        pos = np.asarray(obs["pos"], dtype=float)
        pos_local = pos - origin

        if otype == "box":
            size = np.asarray(obs["size"], dtype=float)
            xyz_min = pos_local - 0.5 * size
            box_obs.append([xyz_min[0], xyz_min[1], xyz_min[2], size[0], size[1], size[2]])
        elif otype in ("sphere", "ball"):
            r = float(obs["radius"])
            ball_obs.append([pos_local[0], pos_local[1], pos_local[2], r])
        elif otype == "cylinder":
            # Env 不支持圆柱，使用包围盒近似
            r = float(obs["radius"])
            h = float(obs["height"])
            size = np.array([2.0 * r, 2.0 * r, h], dtype=float)
            xyz_min = pos_local - 0.5 * size
            box_obs.append([xyz_min[0], xyz_min[1], xyz_min[2], size[0], size[1], size[2]])

    return box_obs, ball_obs


def scene_to_pointcloud(scene_path, output_dir, cfg):
    with open(scene_path, "r") as f:
        scene = json.load(f)

    if "ee_path" not in scene:
        return False

    ws = _scene_workspace(scene, cfg["env_yaml"])
    origin = ws[:3]
    dims = ws[3:] - ws[:3]

    if np.any(dims <= 0):
        raise ValueError(f"workspace_bounds 非法: {ws.tolist()}")

    box_obs, ball_obs = _convert_obstacles_to_env_local(scene, ws)

    env = Env(
        env_dims=tuple(dims.tolist()),
        box_obstacles=box_obs,
        ball_obstacles=ball_obs,
        clearance=0.0,
        resolution=0.1,
    )

    pc_local = generate_rectangle_point_cloud_3d_v1(
        env,
        cfg["n_points"],
        over_sample_scale=cfg["over_sample"],
        use_open3d=True,
    )
    # 转回世界坐标，与 start/goal/ee_path 一致
    pc = pc_local + origin.reshape(1, 3)

    start = np.asarray(scene["start_pos"], dtype=float).reshape(1, 3)
    goal = np.asarray(scene["goal_pos"], dtype=float).reshape(1, 3)
    path_pts = np.asarray(scene["ee_path"], dtype=float)

    masks = {
        "start": get_point_cloud_mask_around_points(pc, start, cfg["start_rad"]),
        "goal": get_point_cloud_mask_around_points(pc, goal, cfg["goal_rad"]),
        "astar": get_point_cloud_mask_around_points(pc, path_pts, cfg["path_rad"]),
    }
    masks["free"] = (1 - masks["start"]) * (1 - masks["goal"])

    token = f"env_{scene['scene_id']:05d}"
    np.savez(
        join(output_dir, f"{token}.npz"),
        pc=pc.astype(np.float32),
        start=masks["start"].astype(np.float32),
        goal=masks["goal"].astype(np.float32),
        astar=masks["astar"].astype(np.float32),
        free=masks["free"].astype(np.float32),
        token=token,
        scene_id=scene["scene_id"],
    )
    return True


def main():
    env_yaml = load_env_config(_ENV_CFG_PATH)

    default_input = _resolve_from_project(env_yaml["environment"].get("output_dir", "data/env"))
    default_output = _resolve_from_project("data/env_pc")

    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=default_input, help="验证通过的场景目录")
    p.add_argument("--output_dir", default=default_output, help="点云输出目录")
    p.add_argument("--n_points", type=int, default=4096)
    p.add_argument("--path_rad", type=float, default=float(env_yaml["collision"].get("path_corridor_radius", 0.3)))
    p.add_argument("--start_rad", type=float, default=float(env_yaml["collision"].get("start_radius", 0.2)))
    p.add_argument("--goal_rad", type=float, default=float(env_yaml["collision"].get("goal_radius", 0.2)))
    p.add_argument("--over_sample", type=int, default=5)
    args = p.parse_args()

    runtime_cfg = {
        "env_yaml": env_yaml,
        "n_points": args.n_points,
        "path_rad": args.path_rad,
        "start_rad": args.start_rad,
        "goal_rad": args.goal_rad,
        "over_sample": args.over_sample,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    scenes = sorted(glob.glob(join(args.input_dir, "env_*.json")))
    print(f"处理 {len(scenes)} 个场景 -> {args.output_dir}")

    success = 0
    for sp in scenes:
        try:
            with open(sp, "r") as f:
                d = json.load(f)
            if "ee_path" not in d:
                print(f"跳过 {os.path.basename(sp)}: 无 ee_path")
                continue
            if scene_to_pointcloud(sp, args.output_dir, runtime_cfg):
                success += 1
        except Exception as e:
            print(f"{os.path.basename(sp)} 失败: {e}")

    print(f"完成: {success}/{len(scenes)} 个场景生成点云")


if __name__ == "__main__":
    main()