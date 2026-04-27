import json, os, glob, numpy as np
from os.path import join
import argparse
import sys

# 添加 src 目录到系统路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ====== 按项目结构调整导入 ======
from planning_utils.env import Env
from generate_datasets_utils.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d_v1
from generate_datasets_utils.point_cloud_mask_utils import get_point_cloud_mask_around_points
# ================================

def scene_to_pointcloud(scene_path, output_dir, cfg):
    with open(scene_path) as f:
        scene = json.load(f)
    
    # 1. 障碍物转换 (适配您的obstacles格式)
    box_obs, ball_obs = [], []
    for obs in scene['obstacles']:
        if obs['type'] == 'box':
            c, s = obs['geometry'][:3], obs['geometry'][3:6]
            box_obs.append([c[0], c[1], c[2], s[0], s[1], s[2]])  # [cx,cy,cz,sx,sy,sz]
        elif obs['type'] in ['ball', 'sphere']:
            c, r = obs['geometry'][:3], obs['geometry'][3]
            ball_obs.append([c[0], c[1], c[2], r])
    
    # 2. 动态计算环境边界 (障碍物包围盒 + 扩展)
    if box_obs or ball_obs:
        pts = []
        for b in box_obs: pts.extend([[b[0]-b[3]/2, b[1]-b[4]/2, b[2]-b[5]/2], [b[0]+b[3]/2, b[1]+b[4]/2, b[2]+b[5]/2]])
        for s in ball_obs: pts.extend([[s[0]-s[3]]*3, [s[0]+s[3]]*3])
        pts = np.array(pts)
        min_b = pts.min(0) - cfg.bound_margin
        max_b = pts.max(0) + cfg.bound_margin
    else:  # 无障碍物时用默认工作空间
        min_b = np.array([-1.5, -1.5, 0.0]) + np.array(scene.get('robot_base_pos', [0,0,0]))
        max_b = np.array([1.5, 1.5, 1.5]) + np.array(scene.get('robot_base_pos', [0,0,0]))
    
    # 3. 构建环境 & 生成点云
    env = Env(
        env_dims=[*min_b, *max_b],
        box_obstacles=box_obs,
        ball_obstacles=ball_obs,
        clearance=0.0,
        resolution=0.1
    )
    pc = generate_rectangle_point_cloud_3d_v1(
        env, cfg.n_points, 
        over_sample_scale=cfg.over_sample, 
        use_open3d=True
    )
    
    # 4. 生成掩码 (关键：使用ee_path!)
    start = np.array(scene['start_pos']).reshape(1,3)
    goal = np.array(scene['goal_pos']).reshape(1,3)
    path_pts = np.array(scene['ee_path'])  # 已保存的末端轨迹
    
    masks = {
        'start': get_point_cloud_mask_around_points(pc, start, cfg.start_rad),
        'goal': get_point_cloud_mask_around_points(pc, goal, cfg.goal_rad),
        'astar': get_point_cloud_mask_around_points(pc, path_pts, cfg.path_rad),
    }
    masks['free'] = (1 - masks['start']) * (1 - masks['goal'])  # 排除起终点邻域
    
    # 5. 保存
    token = f"env_{scene['scene_id']:05d}"
    np.savez(
        join(output_dir, f"{token}.npz"),
        pc=pc.astype(np.float32),
        start=masks['start'].astype(np.float32),
        goal=masks['goal'].astype(np.float32),
        astar=masks['astar'].astype(np.float32),
        free=masks['free'].astype(np.float32),
        token=token,
        scene_id=scene['scene_id']
    )
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', default='data/env', help='验证通过的场景目录')
    p.add_argument('--output_dir', default='data/env_pc', help='点云输出目录')
    p.add_argument('--n_points', type=int, default=4096)
    p.add_argument('--path_rad', type=float, default=0.15)  # 根据机器人尺寸调整
    p.add_argument('--start_rad', type=float, default=0.1)
    p.add_argument('--goal_rad', type=float, default=0.1)
    p.add_argument('--bound_margin', type=float, default=0.3)
    p.add_argument('--over_sample', type=int, default=5)
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    scenes = sorted(glob.glob(join(args.input_dir, 'env_*.json')))
    print(f"处理 {len(scenes)} 个场景 → {args.output_dir}")
    
    success = 0
    for sp in scenes:
        try:
            if 'ee_path' not in json.load(open(sp)):
                print(f"⚠️ 跳过 {os.path.basename(sp)}: 无ee_path (需先运行修改版验证脚本)")
                continue
            if scene_to_pointcloud(sp, args.output_dir, args):
                success += 1
        except Exception as e:
            print(f"❌ {os.path.basename(sp)} 失败: {e}")
    
    print(f"✅ 完成: {success}/{len(scenes)} 个场景生成点云")

if __name__ == '__main__':
    main()