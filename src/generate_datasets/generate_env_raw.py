"""
对 data/env/ 中的每个场景文件执行运动规划验证。
流程: 加载场景 → IK 求解起/终点 → RRT-Connect 规划
规划失败的场景文件将被删除，最后对剩余文件重新编号。

用法:
    python generate_env_raw.py                  # 验证全部场景
    python generate_env_raw.py --start 0 --end 100   # 验证 [0, 100) 范围
"""
import json
import os
import sys
import glob
import argparse
import numpy as np
import yaml
#导入
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "src", "config", "env.yaml")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot_utils.urdf_to_geometry import parse_urdf  , get_ee_link
from robot_utils.compute_kinematic import build_pin_model, compute_fk, compute_ik
from robot_utils.rrt_connect import RRTConnect

#franka joints limits 
JOINT_LIMITS = np.array([
    [-2.9007, 2.9007],
    [-1.8361, 1.8361],
    [-2.9007, 2.9007],
    [-3.0770, -0.1169],
    [-2.8763, 2.8763],
    [0.4398, 4.6216],
    [-3.0508, 3.0508],
])

#get end pose 


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"配置文件为空或格式错误: {config_path}")
    return cfg


def _solve_ik_with_checks(model, data, ee_link, target_world_pos, base_pos, collision_checker, n_starts=50, tol=0.01):
    """
    使用新 IK 接口做多起点求解，并保留碰撞与误差校验。
    返回 (q, success)。当无法得到候选解时返回 (None, False)。
    """
    target_world_pos = np.asarray(target_world_pos, dtype=float).reshape(3)
    base_offset = np.asarray(base_pos, dtype=float).reshape(3)
    target_local = target_world_pos - base_offset
    target_rpy = np.zeros(3, dtype=float)

    best_q = None
    best_err = np.inf

    for i in range(n_starts):
        if i == 0:
            q0 = np.mean(JOINT_LIMITS, axis=1)
        else:
            q0 = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        ik_result = compute_ik(
            model=model,
            data=data,
            ee_frame_name=ee_link,
            target_xyz=target_local,
            target_rpy=target_rpy,
            q0=q0,
            max_iter=200,
            tol=tol,
            with_orientation=False,
        )

        q = np.asarray(ik_result["q"], dtype=float).reshape(-1)
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        if not collision_checker(q):
            continue

        fk = compute_fk(model, data, q, ee_link)
        ee_world = fk["xyz"] + base_offset
        err = np.linalg.norm(ee_world - target_world_pos)

        if err < best_err:
            best_err = err
            best_q = q

        if ik_result["success"] and err <= 0.05:
            return q, True

    if best_q is None:
        return None, False
    return best_q, False


def validate_scene(model, pin_data, links, joints, ee_link, scene_path, planner_clearance):
    """
    验证单个场景是否可规划。
    返回 True 表示规划成功，False 表示应删除。
    """
    with open(scene_path, 'r') as f:
        scene_data = json.load(f)

    obstacles = scene_data['obstacles']
    start_pos = np.array(scene_data['start_pos'])
    goal_pos = np.array(scene_data['goal_pos'])
    base_pos = tuple(scene_data.get('robot_base_pos', [0.0, 0.0, 0.0]))

    revolute = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    n_dof = len(revolute)

    # 构建碰撞检查器
    col_checker = RRTConnect(
        np.zeros(n_dof), np.zeros(n_dof),
        step_len=0.1, iter_max=1,
        links=links, joints=joints, obstacles=obstacles,
        clearance=planner_clearance, base_pos=base_pos, joint_limits=JOINT_LIMITS,
    )

    # IK 求解起点
    q_start, ok_s = _solve_ik_with_checks(
        model=model,
        data=pin_data,
        ee_link=ee_link,
        target_world_pos=start_pos,
        base_pos=base_pos,
        collision_checker=col_checker.is_collision_free,
        n_starts=50,
        tol=0.01,
    )
    if not ok_s:
        if q_start is None:
            return False
        fk = compute_fk(model, pin_data, q_start, ee_link)
        err = np.linalg.norm((fk['xyz'] + np.asarray(base_pos, dtype=float)) - start_pos)
        if err > 0.05:
            return False

    # IK 求解终点
    q_goal, ok_g = _solve_ik_with_checks(
        model=model,
        data=pin_data,
        ee_link=ee_link,
        target_world_pos=goal_pos,
        base_pos=base_pos,
        collision_checker=col_checker.is_collision_free,
        n_starts=50,
        tol=0.01,
    )
    if not ok_g:
        if q_goal is None:
            return False
        fk = compute_fk(model, pin_data, q_goal, ee_link)
        err = np.linalg.norm((fk['xyz'] + np.asarray(base_pos, dtype=float)) - goal_pos)
        if err > 0.05:
            return False

    # RRT-Connect 规划
    planner = RRTConnect(
        q_start, q_goal,
        step_len=0.1, iter_max=5000,
        links=links, joints=joints, obstacles=obstacles,
        clearance=planner_clearance, base_pos=base_pos, joint_limits=JOINT_LIMITS,
    )
    path = planner.planning()
    if path is None or len(path) == 0:
        return False

    # 规划成功后，保存末端轨迹到场景文件
    ee_path = []
    for q in path:
        fk = compute_fk(model, pin_data, q, ee_link)
        ee_pos = fk['xyz'] + np.asarray(base_pos, dtype=float)
        ee_path.append(ee_pos.tolist())

    scene_data['ee_path'] = ee_path
    with open(scene_path, 'w') as f:
        json.dump(scene_data, f, indent=2)

    return True


def renumber_scenes(data_dir):
    """将剩余的场景文件按顺序重新编号"""
    files = sorted(glob.glob(os.path.join(data_dir, "env_*.json")))
    for new_id, old_path in enumerate(files):
        new_name = os.path.join(data_dir, f"env_{new_id:05d}.json")
        if old_path != new_name:
            # 更新文件内的 scene_id
            with open(old_path, 'r') as f:
                data = json.load(f)
            data['scene_id'] = new_id
            with open(old_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.rename(old_path, new_name)
    print(f"重新编号完成，共 {len(files)} 个场景")


def main():
    
    parser = argparse.ArgumentParser(description="验证场景可规划性，删除不可规划的场景")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    parser.add_argument('--start', type=int, default=None, help="起始场景 ID（含）")
    parser.add_argument('--end', type=int, default=None, help="结束场景 ID（不含）")
    parser.add_argument('--no-renumber', action='store_true', help="不重新编号")
    args = parser.parse_args()

    config = load_config(args.config)
    env_cfg = config.get('environment', {})
    robot_cfg = config.get('robot', {})
    col_cfg = config.get('collision', {})
    generation_cfg = config.get('generation', {})

    data_dir_cfg = env_cfg.get('output_dir', 'data/env')
    data_dir = data_dir_cfg if os.path.isabs(data_dir_cfg) else os.path.join(_PROJECT_ROOT, data_dir_cfg)

    urdf_cfg = robot_cfg.get('urdf_path', 'robots/franka_panda_gem.urdf')
    urdf_path = urdf_cfg if os.path.isabs(urdf_cfg) else os.path.join(_PROJECT_ROOT, 'src', urdf_cfg)

    planner_clearance = float(col_cfg.get('clearance', 0.05))
    progress_interval = int(generation_cfg.get('progress_interval', 50))

    # 收集场景文件
    all_files = sorted(glob.glob(os.path.join(data_dir, "env_*.json")))
    if not all_files:
        print("没有找到场景文件"); return

    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end or len(all_files)
        all_files = all_files[start:end]

    print(f"共 {len(all_files)} 个场景待验证")

    # 解析 URDF（只需一次）
    links, joints = parse_urdf(urdf_path)
    ee_link = get_ee_link(joints)
    pin_model, pin_data = build_pin_model(urdf_path)
    if pin_model.nq != JOINT_LIMITS.shape[0]:
        raise ValueError(
            f"Pinocchio 模型自由度({pin_model.nq})与 JOINT_LIMITS({JOINT_LIMITS.shape[0]})不一致"
        )
    print(f"URDF 加载完成，ee_link={ee_link}")


    success_count = 0
    fail_count = 0
    deleted_files = []

    for i, scene_path in enumerate(all_files):
        scene_name = os.path.basename(scene_path)
        try:
            ok = validate_scene(pin_model, pin_data, links, joints, ee_link, scene_path, planner_clearance)
        except Exception as e:
            print(f"  [{i+1}/{len(all_files)}] {scene_name}: 异常 - {e}")
            ok = False

        if ok:
            success_count += 1
            if progress_interval <= 0 or ((i + 1) % progress_interval == 0):
                print(f"  [{i+1}/{len(all_files)}] {scene_name}: 通过")
        else:
            fail_count += 1
            print(f"  [{i+1}/{len(all_files)}] {scene_name}: 失败，删除")
            deleted_files.append(scene_path)
            os.remove(scene_path)

    print(f"\n验证完成: 通过 {success_count}, 失败 {fail_count} (已删除)")

    # 重新编号
    if not args.no_renumber and fail_count > 0:
        renumber_scenes(data_dir)


if __name__ == "__main__":
    main()
