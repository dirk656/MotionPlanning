import json
import numpy as np
import os
import sys
import yaml

config_path = "/config/env.yaml"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from planning_utils.collision_check_utils import points_in_AABB_3d, points_in_ball_3d, points_in_cylinder_3d, points_in_robot_arm_3d
from robot_utils.urdf_to_geometry import parse_urdf, get_robot_collision_bodies
from robot_utils.fk_solver import compute_fk


def check_point_in_obstacles(point, obstacles, clearance=0.0):
    """使用 collision_check_utils 检查单个 3D 点是否在任意障碍物内（含安全间距）"""
    point_tuple = tuple(point.tolist())
    boxes, spheres, cylinders = [], [], []
    for obs in obstacles:
        pos = np.array(obs['pos'])
        if obs['type'] == 'box':
            size = np.array(obs['size'])
            min_corner = pos - size / 2  # 中心点转最小角点
            boxes.append([*min_corner, *size])
        elif obs['type'] == 'sphere':
            spheres.append([*pos, obs['radius']])
        elif obs['type'] == 'cylinder':
            cylinders.append([*pos, obs['radius'], obs['height']])
    box_arr = np.array(boxes) if boxes else None
    sphere_arr = np.array(spheres) if spheres else None
    cylinder_arr = np.array(cylinders) if cylinders else None
    in_box = points_in_AABB_3d(point_tuple, box_arr, clearance=clearance)
    in_sphere = points_in_ball_3d(point_tuple, sphere_arr, clearance=clearance)
    in_cylinder = points_in_cylinder_3d(point_tuple, cylinder_arr, clearance=clearance)
    return in_box or in_sphere or in_cylinder


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f" '{config_path}' 未找到")
        exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_scene(scene_id, config, robot_collision_bodies=None):
    # 1. 从配置中提取环境边界
    env_cfg = config['environment']
    workspace_bounds = np.array(env_cfg['workspace_bounds'])
    min_bounds = workspace_bounds[:3]
    max_bounds = workspace_bounds[3:]
    
    
    # 2. 从配置中提取障碍物参数
    obs_cfg = config['obstacles']
    num_obstacles = obs_cfg['count']

    # 碰撞安全区
    col_cfg = config.get('collision', {})
    start_radius = col_cfg.get('start_radius', 0.5)
    goal_radius = col_cfg.get('goal_radius', 0.5)
    min_start_goal_distance = col_cfg.get('min_start_goal_distance', 2.0)
    max_start_goal_attempt = col_cfg.get('max_start_goal_attempt', 200)
    robot_clearance = col_cfg.get('robot_clearance', 0.1)
    boundary_margin = col_cfg.get('boundary_margin', 0.05)
    sg_min = min_bounds + boundary_margin  # 起终点采样的收缩边界
    sg_max = max_bounds - boundary_margin

    # 先生成起终点，确保距离大于 min_start_goal_distance 且不在机械臂内
    for _ in range(max_start_goal_attempt):
        start_pos = np.random.uniform(sg_min, sg_max)
        goal_pos = np.random.uniform(sg_min, sg_max)
        if np.sum((start_pos - goal_pos) ** 2) <= min_start_goal_distance ** 2:
            continue
        if robot_collision_bodies:
            s_in = points_in_robot_arm_3d(tuple(start_pos.tolist()), robot_collision_bodies, clearance=robot_clearance)
            g_in = points_in_robot_arm_3d(tuple(goal_pos.tolist()), robot_collision_bodies, clearance=robot_clearance)
            if s_in or g_in:
                continue
        break
    else:
        print(f"警告: 场景 {scene_id} 起终点采样 {max_start_goal_attempt} 次仍不满足约束")
    # 路径管道：保证起终点连线附近有障碍物
    path_corridor_radius = col_cfg.get('path_corridor_radius', 2.0)
    min_path_obstacles = col_cfg.get('min_path_obstacles', 2)

    def point_to_segment_dist(p, a, b):
        """点 p 到线段 ab 的最短距离"""
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-8), 0, 1)
        return np.linalg.norm(p - (a + t * ab))

    def sample_pos(near_path=False):
        """采样一个满足安全区约束的位置，near_path=True 时约束在管道内"""
        for _ in range(200):
            if near_path:
                # 在起终点连线上随机插值一个点，然后加扰动
                t = np.random.uniform(0.1, 0.9)
                center = start_pos + t * (goal_pos - start_pos)
                pos = center + np.random.normal(0, path_corridor_radius / 3, size=3)
                pos = np.clip(pos, min_bounds, max_bounds)
            else:
                pos = np.random.uniform(min_bounds, max_bounds)
            if (np.linalg.norm(pos - start_pos) > start_radius and
                    np.linalg.norm(pos - goal_pos) > goal_radius):
                if not near_path or point_to_segment_dist(pos, start_pos, goal_pos) < path_corridor_radius:
                    # 检查是否与机械臂碰撞
                    if robot_collision_bodies:
                        if points_in_robot_arm_3d(tuple(pos.tolist()), robot_collision_bodies, clearance=robot_clearance):
                            continue
                    return pos
        return pos  # fallback

    obstacles = []

    # 先生成管道内的障碍物
    path_count = min(min_path_obstacles, num_obstacles)
    remaining_count = num_obstacles - path_count
    
    for i in range(num_obstacles):
        near_path = (i < path_count)
        
        # 随机选择形状类型
        shape_options = []
        if obs_cfg['box'].get('enabled', True): shape_options.append(0)
        if obs_cfg['sphere'].get('enabled', True): shape_options.append(1)
        if obs_cfg['cylinder'].get('enabled', True): shape_options.append(2)
        
        if not shape_options:
            raise ValueError("配置文件中至少需要启用一种障碍物类型。")

        shape_type = np.random.choice(shape_options)
        pos = sample_pos(near_path=near_path)
        
        if shape_type == 0: # Box
            b_cfg = obs_cfg['box']
            size = np.random.uniform(b_cfg['size_min'], b_cfg['size_max'])
            obstacles.append({'type': 'box', 'pos': pos.tolist(), 'size': size.tolist()})
        elif shape_type == 1: # Sphere
            s_cfg = obs_cfg['sphere']
            r = np.random.uniform(s_cfg['radius_min'], s_cfg['radius_max'])
            obstacles.append({'type': 'sphere', 'pos': pos.tolist(), 'radius': float(r)})
        elif shape_type == 2: # Cylinder
            c_cfg = obs_cfg['cylinder']
            r = np.random.uniform(c_cfg['radius_min'], c_cfg['radius_max'])
            h = np.random.uniform(c_cfg['height_min'], c_cfg['height_max'])
            obstacles.append({'type': 'cylinder', 'pos': pos.tolist(), 'radius': float(r), 'height': float(h)})

    # 碰撞检测：确保起终点不在任何障碍物内（含安全间距）
    collision_clearance = col_cfg.get('clearance', 0.1)
    max_resample = col_cfg.get('max_resample', 200)

    for _ in range(max_resample):
        in_obs = check_point_in_obstacles(start_pos, obstacles, collision_clearance)
        in_arm = points_in_robot_arm_3d(tuple(start_pos.tolist()), robot_collision_bodies, clearance=robot_clearance) if robot_collision_bodies else False
        if not in_obs and not in_arm:
            break
        start_pos = np.random.uniform(sg_min, sg_max)
    else:
        print(f"警告: 场景 {scene_id} 起点重采样 {max_resample} 次仍在障碍物/机械臂内")

    for _ in range(max_resample):
        in_obs = check_point_in_obstacles(goal_pos, obstacles, collision_clearance)
        in_arm = points_in_robot_arm_3d(tuple(goal_pos.tolist()), robot_collision_bodies, clearance=robot_clearance) if robot_collision_bodies else False
        if not in_obs and not in_arm:
            break
        goal_pos = np.random.uniform(sg_min, sg_max)
    else:
        print(f"警告: 场景 {scene_id} 终点重采样 {max_resample} 次仍在障碍物/机械臂内")

    data = {
        'scene_id': scene_id,
        'workspace_bounds': workspace_bounds.tolist(),
        'obstacles': obstacles,
        'start_pos': start_pos.tolist(),
        'goal_pos': goal_pos.tolist()
    }
    
    # ---------------------------------------------------------
    # 【核心修改】自动创建 data/env 文件夹逻辑
    # ---------------------------------------------------------
    output_dir = os.path.join(project_root, env_cfg['output_dir'])
    
    # os.makedirs 会递归创建所有不存在的父目录
    # exist_ok=True 确保如果目录已存在也不会报错
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"已自动创建: {output_dir}")
        except OSError as e:
            print(f"创建目录失败: {e}")
            sys.exit(1)
    # ---------------------------------------------------------

    filename = os.path.join(output_dir, f"env_{scene_id:05d}.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return filename

if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'env.yaml')
    config_file = default_config
    
    if len(sys.argv) > 2:
        config_file = sys.argv[2]
    
    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"配置加载失败: {e}")
        sys.exit(1)

    if config is None:
        print(f"配置文件为空或格式错误: {config_file}")
        sys.exit(1)

    # 设置随机种子
    seed = config.get('random_seed')
    if seed is not None:
        np.random.seed(seed)

    # 初始化机械臂碰撞体
    robot_cfg = config.get('robot', {})
    urdf_path = robot_cfg.get('urdf_path')
    robot_collision_bodies = None
    if urdf_path:
        if os.path.isabs(urdf_path):
            urdf_full = urdf_path
        else:
            # URDF 路径相对于 config 文件所在目录的父目录（即 src/）
            urdf_full = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(config_file)), '..', urdf_path)
            )
        bounds = config['environment']['workspace_bounds']
        base_pos = robot_cfg.get('base_position', [
            (bounds[0] + bounds[3]) / 2,
            (bounds[1] + bounds[4]) / 2,
            bounds[2],
        ])
        links, joints = parse_urdf(urdf_full)
        _, _, world_collisions = compute_fk(links, joints, base_pos=tuple(base_pos))
        robot_collision_bodies = get_robot_collision_bodies(world_collisions)
        print(f"已加载机械臂碰撞体: {len(robot_collision_bodies)} 个")

    if len(sys.argv) > 1:
        try:
            sid = int(sys.argv[1])
            file_path = generate_scene(sid, config, robot_collision_bodies)
            print(f"已生成: {file_path}")
        except ValueError:
            print("错误: Scene ID 必须是整数。")
            sys.exit(1)
    else:
        count = config['environment']['default_count']
        output_dir = config['environment']['output_dir']
        
        print(f"🚀 开始批量生成 {count} 个场景...")
        print(f"📂 目标目录: {output_dir} (若不存在将自动创建)")
        print(f"📏 环境大小: {config['environment']['workspace_bounds']}")
        
        for i in range(count):
            generate_scene(i, config, robot_collision_bodies)
            if (i + 1) % 50 == 0:
                print(f"   进度: {i+1}/{count}")
        
        print(f"✅ 完成！所有文件已保存至: {os.path.abspath(output_dir)}")