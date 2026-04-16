# pyright: reportMissingImports=false
import argparse
import glob
import json
import os
import sys
from os.path import join

import numpy as np
import yaml


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ENV_CFG_PATH = os.path.join(_PROJECT_ROOT, "src", "config", "env.yaml")
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")

if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from generate_datasets_utils.point_cloud_mask_utils import get_point_cloud_mask_around_points
from generate_datasets_utils.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d_v1
from planning_utils.env import Env


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


def scene_to_sample(scene_path, cfg):
    with open(scene_path, "r") as f:
        scene = json.load(f)

    if "ee_path" not in scene:
        return None

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
    return {
        "pc": pc.astype(np.float32),
        "start": np.asarray(masks["start"], dtype=np.float32).reshape(-1),
        "goal": np.asarray(masks["goal"], dtype=np.float32).reshape(-1),
        "free": np.asarray(masks["free"], dtype=np.float32).reshape(-1),
        "astar": np.asarray(masks["astar"], dtype=np.float32).reshape(-1),
        "token": token,
    }


def _save_split_npz(out_path, samples):
    if len(samples) == 0:
        np.savez(
            out_path,
            pc=np.empty((0, 0, 3), dtype=np.float32),
            start=np.empty((0, 0), dtype=np.float32),
            goal=np.empty((0, 0), dtype=np.float32),
            free=np.empty((0, 0), dtype=np.float32),
            astar=np.empty((0, 0), dtype=np.float32),
            token=np.empty((0,), dtype="U1"),
        )
        return

    pc = np.stack([s["pc"] for s in samples], axis=0).astype(np.float32)
    start = np.stack([s["start"] for s in samples], axis=0).astype(np.float32)
    goal = np.stack([s["goal"] for s in samples], axis=0).astype(np.float32)
    free = np.stack([s["free"] for s in samples], axis=0).astype(np.float32)
    astar = np.stack([s["astar"] for s in samples], axis=0).astype(np.float32)
    token = np.asarray([s["token"] for s in samples])

    np.savez(out_path, pc=pc, start=start, goal=goal, free=free, astar=astar, token=token)


def main():
    env_yaml = load_env_config(_ENV_CFG_PATH)

    default_input = _resolve_from_project(env_yaml["environment"].get("output_dir", "data/env"))
    default_output = _resolve_from_project("data/random_3d")

    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=default_input, help="验证通过的场景目录")
    p.add_argument("--output_dir", default=default_output, help="训练数据目录，输出 train.npz/val.npz")
    p.add_argument("--n_points", type=int, default=2048)
    p.add_argument("--path_rad", type=float, default=float(env_yaml["collision"].get("path_corridor_radius", 0.3)))
    p.add_argument("--start_rad", type=float, default=float(env_yaml["collision"].get("start_radius", 0.2)))
    p.add_argument("--goal_rad", type=float, default=float(env_yaml["collision"].get("goal_radius", 0.2)))
    p.add_argument("--over_sample", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.9, help="训练集比例")
    p.add_argument("--seed", type=int, default=42, help="切分随机种子")
    args = p.parse_args()

    runtime_cfg = {
        "env_yaml": env_yaml,
        "n_points": args.n_points,
        "path_rad": args.path_rad,
        "start_rad": args.start_rad,
        "goal_rad": args.goal_rad,
        "over_sample": args.over_sample,
    }

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio 必须在 (0,1) 之间")

    os.makedirs(args.output_dir, exist_ok=True)
    scenes = sorted(glob.glob(join(args.input_dir, "env_*.json")))
    print(f"处理 {len(scenes)} 个场景 -> {args.output_dir}")

    samples = []
    skipped = 0
    for sp in scenes:
        try:
            with open(sp, "r") as f:
                d = json.load(f)
            if "ee_path" not in d:
                print(f"跳过 {os.path.basename(sp)}: 无 ee_path")
                skipped += 1
                continue
            sample = scene_to_sample(sp, runtime_cfg)
            if sample is not None:
                samples.append(sample)
        except Exception as e:
            print(f"{os.path.basename(sp)} 失败: {e}")

    if len(samples) == 0:
        raise RuntimeError("没有可用样本，无法生成 train/val 数据")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    split_idx = int(len(indices) * args.train_ratio)
    split_idx = max(1, min(split_idx, len(indices) - 1))

    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]

    train_path = join(args.output_dir, "train.npz")
    val_path = join(args.output_dir, "val.npz")
    _save_split_npz(train_path, train_samples)
    _save_split_npz(val_path, val_samples)

    print(
        f"完成: 可用 {len(samples)} / {len(scenes)} (跳过 {skipped}) | "
        f"train={len(train_samples)}, val={len(val_samples)}"
    )
    print(f"已保存: {train_path}")
    print(f"已保存: {val_path}")


if __name__ == "__main__":
    main()