import json
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

# 配置
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "env")
# 默认查看最新的一个文件，或者通过命令行指定 ID
DEFAULT_SCENE_ID =  2

def load_scene(scene_id):
    filename = os.path.join(DATA_DIR, f"env_{scene_id:05d}.json")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到场景文件: {filename}")
    
    with open(filename, 'r') as f:
        return json.load(f)

def plot_cube(ax, center, size, color='blue', alpha=0.6):
    """绘制长方体 (Box)"""
    x, y, z = center
    dx, dy, dz = size
    
    # 计算顶点
    vertices = [
        [[x-dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z-dz/2], [x+dx/2, y+dy/2, z-dz/2], [x-dx/2, y+dy/2, z-dz/2]],
        [[x-dx/2, y-dy/2, z+dz/2], [x+dx/2, y-dy/2, z+dz/2], [x+dx/2, y+dy/2, z+dz/2], [x-dx/2, y+dy/2, z+dz/2]],
        [[x-dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z+dz/2], [x-dx/2, y-dy/2, z+dz/2]],
        [[x-dx/2, y+dy/2, z-dz/2], [x+dx/2, y+dy/2, z-dz/2], [x+dx/2, y+dy/2, z+dz/2], [x-dx/2, y+dy/2, z+dz/2]],
        [[x-dx/2, y-dy/2, z-dz/2], [x-dx/2, y+dy/2, z-dz/2], [x-dx/2, y+dy/2, z+dz/2], [x-dx/2, y-dy/2, z+dz/2]],
        [[x+dx/2, y-dy/2, z-dz/2], [x+dx/2, y+dy/2, z-dz/2], [x+dx/2, y+dy/2, z+dz/2], [x+dx/2, y-dy/2, z+dz/2]]
    ]
    
    poly = Poly3DCollection(vertices, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha)
    ax.add_collection3d(poly)

def plot_sphere(ax, center, radius, color='green', alpha=0.6):
    """绘制球体 (Sphere)"""
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def plot_cylinder(ax, center, radius, height, color='orange', alpha=0.6):
    """绘制圆柱体 (Cylinder)"""
    z = np.linspace(center[2] - height/2, center[2] + height/2, 20)
    theta = np.linspace(0, 2*np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)
    
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)

def visualize_scene(scene_id):
    data = load_scene(scene_id)
    
    obstacles = data['obstacles']
    start_pos = data['start_pos']
    goal_pos = data['goal_pos']
    bounds = data.get('workspace_bounds', [-5, -5, -5, 5, 5, 5])
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制工作空间边界 (线框)
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    # 绘制一个简单的边界框逻辑 (仅绘制边线以节省性能)
    edges = [
        [[x_min, y_min, z_min], [x_max, y_min, z_min]], [[x_min, y_max, z_min], [x_max, y_max, z_min]],
        [[x_min, y_min, z_max], [x_max, y_min, z_max]], [[x_min, y_max, z_max], [x_max, y_max, z_max]],
        [[x_min, y_min, z_min], [x_min, y_max, z_min]], [[x_max, y_min, z_min], [x_max, y_max, z_min]],
        [[x_min, y_min, z_max], [x_min, y_max, z_max]], [[x_max, y_min, z_max], [x_max, y_max, z_max]],
        [[x_min, y_min, z_min], [x_min, y_min, z_max]], [[x_max, y_min, z_min], [x_max, y_min, z_max]],
        [[x_min, y_max, z_min], [x_min, y_max, z_max]], [[x_max, y_max, z_min], [x_max, y_max, z_max]]
    ]
    for edge in edges:
        xs, ys, zs = zip(*edge)
        ax.plot(xs, ys, zs, 'k--', linewidth=1, alpha=0.3)

    # 2. 绘制障碍物
    for obs in obstacles:
        o_type = obs['type']
        pos = obs['pos']
        
        if o_type == 'box':
            plot_cube(ax, pos, obs['size'], color='#FF6B6B', alpha=0.7)
        elif o_type == 'sphere':
            plot_sphere(ax, pos, obs['radius'], color='#4ECDC4', alpha=0.7)
        elif o_type == 'cylinder':
            plot_cylinder(ax, pos, obs['radius'], obs['height'], color='#FFE66D', alpha=0.7)

    # 3. 绘制起点和终点
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], c='lime', s=100, marker='o', label='Start', edgecolors='black')
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], c='red', s=100, marker='x', label='Goal', linewidths=2)
    
    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Scene ID: {scene_id:05d}')
    ax.legend()
    
    # 设置坐标轴范围 (保持比例一致，防止变形)
    max_range = np.array([
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ]).max() / 2.0
    
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置视角
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    scene_id = DEFAULT_SCENE_ID
    
    if len(sys.argv) > 1:
        try:
            scene_id = int(sys.argv[1])
        except ValueError:
            print("用法: python 02_visualize_env.py [scene_id]")
            print("如果不提供 ID，默认尝试加载最新的或 ID 为 0 的场景。")
            # 如果没有提供 ID，尝试找最大的 ID 或者默认为 0
            if not os.path.exists(DATA_DIR):
                print(f"错误: 目录 {DATA_DIR} 不存在。请先生成环境。")
                sys.exit(1)
            files = [f for f in os.listdir(DATA_DIR) if f.startswith('env_') and f.endswith('.json')]
            if files:
                # 提取最大的 ID
                ids = [int(f.split('_')[1].split('.')[0]) for f in files]
                scene_id = max(ids)
                print(f"未指定 ID，自动加载最新生成的场景: {scene_id}")
            else:
                scene_id = 0
    else:
        # 没给参数，默认找最新的
        if os.path.exists(DATA_DIR):
            files = [f for f in os.listdir(DATA_DIR) if f.startswith('env_') and f.endswith('.json')]
            if files:
                ids = [int(f.split('_')[1].split('.')[0]) for f in files]
                scene_id = max(ids)
                print(f"未指定 ID，自动加载最新生成的场景: {scene_id}")
            else:
                scene_id = 0
        else:
            scene_id = 0

    try:
        visualize_scene(scene_id)
    except FileNotFoundError as e:
        print(e)
        print("提示: 请先运行生成脚本 (python 01_generate_env.py)")
    except Exception as e:
        print(f"发生错误: {e}")