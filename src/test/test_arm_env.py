
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "env")
URDF_PATH = os.path.join(_PROJECT_ROOT, "src", "config", "franka_panda_gem.urdf")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from planning_utils.urdf_to_python import parse_urdf, compute_fk


# ───────── 绘图函数 ─────────

def plot_cube(ax, center, size, color='blue', alpha=0.6):
    x, y, z = center
    dx, dy, dz = size
    vertices = [
        [[x-dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z-dz/2], [x+dx/2, y+dy/2, z-dz/2], [x-dx/2, y+dy/2, z-dz/2]],
        [[x-dx/2, y-dy/2, z+dz/2], [x+dx/2, y-dy/2, z+dz/2], [x+dx/2, y+dy/2, z+dz/2], [x-dx/2, y+dy/2, z+dz/2]],
        [[x-dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z-dz/2], [x+dx/2, y-dy/2, z+dz/2], [x-dx/2, y-dy/2, z+dz/2]],
        [[x-dx/2, y+dy/2, z-dz/2], [x+dx/2, y+dy/2, z-dz/2], [x+dx/2, y+dy/2, z+dz/2], [x-dx/2, y+dy/2, z+dz/2]],
        [[x-dx/2, y-dy/2, z-dz/2], [x-dx/2, y+dy/2, z-dz/2], [x-dx/2, y+dy/2, z+dz/2], [x-dx/2, y-dy/2, z+dz/2]],
        [[x+dx/2, y-dy/2, z-dz/2], [x+dx/2, y+dy/2, z-dz/2], [x+dx/2, y+dy/2, z+dz/2], [x+dx/2, y-dy/2, z+dz/2]],
    ]
    poly = Poly3DCollection(vertices, facecolors=color, linewidths=0.5, edgecolors='black', alpha=alpha)
    ax.add_collection3d(poly)


def plot_sphere_surf(ax, center, radius, color='green', alpha=0.6):
    u, v = np.mgrid[0:2*np.pi:16j, 0:np.pi:8j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_cylinder_surf(ax, center, radius, length, rotation_matrix, color='orange', alpha=0.6):
    """绘制沿任意方向的圆柱体，中心在 center，轴由 rotation_matrix 的 z 轴决定"""
    t = np.linspace(-length / 2, length / 2, 15)
    theta = np.linspace(0, 2 * np.pi, 20)
    T, Theta = np.meshgrid(t, theta)

    # 在圆柱体局部坐标系中生成点 (x_local = r*cos, y_local = r*sin, z_local = t)
    X_local = radius * np.cos(Theta)
    Y_local = radius * np.sin(Theta)
    Z_local = T

    # 将局部坐标转换到世界坐标
    pts_local = np.stack([X_local.ravel(), Y_local.ravel(), Z_local.ravel()], axis=1)  # (N, 3)
    pts_world = (rotation_matrix @ pts_local.T).T + center  # (N, 3)

    X = pts_world[:, 0].reshape(X_local.shape)
    Y = pts_world[:, 1].reshape(Y_local.shape)
    Z = pts_world[:, 2].reshape(Z_local.shape)

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def plot_robot_visuals(ax, world_visuals):
    """将 URDF 的 visual 几何在 3D 轴上绘制"""
    for vis in world_visuals:
        geom = vis['geometry']
        T = vis['transform']
        pos = T[:3, 3]
        R = T[:3, :3]
        gtype = geom.get('type')

        if gtype == 'sphere':
            r = float(geom['radius'])
            plot_sphere_surf(ax, pos, r, color='#FFA500', alpha=0.9)
        elif gtype == 'cylinder':
            r = float(geom['radius'])
            l = float(geom['length'])
            plot_cylinder_surf(ax, pos, r, l, R, color='#C0C0C0', alpha=0.8)


def plot_env_obstacles(ax, obstacles):
    for obs in obstacles:
        pos = obs['pos']
        if obs['type'] == 'box':
            plot_cube(ax, pos, obs['size'], color='#FF6B6B', alpha=0.5)
        elif obs['type'] == 'sphere':
            plot_sphere_surf(ax, pos, obs['radius'], color='#4ECDC4', alpha=0.5)
        elif obs['type'] == 'cylinder':
            plot_cylinder_surf(ax, pos, obs['radius'], obs['height'], np.eye(3), color='#FFE66D', alpha=0.5)


# ───────── 主函数 ─────────

def load_scene(scene_id):
    filename = os.path.join(DATA_DIR, f"env_{scene_id:05d}.json")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到场景文件: {filename}")
    with open(filename, 'r') as f:
        return json.load(f)


def main(scene_id=0):
    # 1. 加载环境
    data = load_scene(scene_id)
    bounds = data['workspace_bounds']
    obstacles = data['obstacles']
    start_pos = data['start_pos']
    goal_pos = data['goal_pos']

    # 2. 解析 URDF 并计算正运动学（零位姿态），机械臂基座在环境底面中心
    x_mid = (bounds[0] + bounds[3]) / 2
    y_mid = (bounds[1] + bounds[4]) / 2
    z_base = bounds[2]  # 底面 z=0
    base_pos = (x_mid, y_mid, z_base)

    links, joints = parse_urdf(URDF_PATH)
    world_visuals, link_transforms, _ = compute_fk(links, joints, base_pos=base_pos)

    # 3. 绘图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 工作空间边界线框
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    edges = [
        [[x_min, y_min, z_min], [x_max, y_min, z_min]], [[x_min, y_max, z_min], [x_max, y_max, z_min]],
        [[x_min, y_min, z_max], [x_max, y_min, z_max]], [[x_min, y_max, z_max], [x_max, y_max, z_max]],
        [[x_min, y_min, z_min], [x_min, y_max, z_min]], [[x_max, y_min, z_min], [x_max, y_max, z_min]],
        [[x_min, y_min, z_max], [x_min, y_max, z_max]], [[x_max, y_min, z_max], [x_max, y_max, z_max]],
        [[x_min, y_min, z_min], [x_min, y_min, z_max]], [[x_max, y_min, z_min], [x_max, y_min, z_max]],
        [[x_min, y_max, z_min], [x_min, y_max, z_max]], [[x_max, y_max, z_min], [x_max, y_max, z_max]],
    ]
    for edge in edges:
        xs, ys, zs = zip(*edge)
        ax.plot(xs, ys, zs, 'k--', linewidth=0.8, alpha=0.3)

    # 障碍物
    plot_env_obstacles(ax, obstacles)

    # 起终点
    ax.scatter(*start_pos, c='lime', s=120, marker='o', label='Start', edgecolors='black', zorder=5)
    ax.scatter(*goal_pos, c='red', s=120, marker='X', label='Goal', linewidths=2, zorder=5)

    # 机械臂
    plot_robot_visuals(ax, world_visuals)

    # 机械臂基座标记
    ax.scatter(*base_pos, c='blue', s=80, marker='s', label='Robot Base', edgecolors='black', zorder=5)

    # 连接关节中心的骨架线
    joint_positions = []
    for jnt in joints:
        child_T = link_transforms.get(jnt['child'])
        if child_T is not None:
            joint_positions.append(child_T[:3, 3])
    if joint_positions:
        jp = np.array(joint_positions)
        ax.plot(jp[:, 0], jp[:, 1], jp[:, 2], 'b-o', linewidth=1.5, markersize=3, alpha=0.7, label='Arm skeleton')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Franka Panda in Scene {scene_id:05d}')
    ax.legend(loc='upper left', fontsize=8)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid = [(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.view_init(elev=25, azim=-50)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sid = 0
    if len(sys.argv) > 1:
        sid = int(sys.argv[1])
    main(sid)
