"""
测试机械臂从场景起点运动到终点。
流程: 加载场景 → IK 求解 → RRT-Connect 规划 → 3D 动画可视化
用法: python test_arm.py [scene_id]
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "env")
URDF_PATH = os.path.join(_PROJECT_ROOT, "src", "robots", "franka_panda_gem.urdf")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot_utils.urdf_to_geometry import parse_urdf
from robot_utils.fk_solver import compute_fk
from robot_utils.ik_solver import solve_ik_multi_start
from robot_utils.rrt_connect import RRTConnect, shortcut_path

# ───────── Franka Panda 关节限位 ─────────
JOINT_LIMITS = np.array([
    [-2.9007, 2.9007],
    [-1.8361, 1.8361],
    [-2.9007, 2.9007],
    [-3.0770, -0.1169],
    [-2.8763, 2.8763],
    [0.4398, 4.6216],
    [-3.0508, 3.0508],
])

# ───────── 绘图工具 ─────────

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
    u, v = np.mgrid[0:2*np.pi:12j, 0:np.pi:6j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_cylinder_surf(ax, center, radius, length, rotation_matrix, color='orange', alpha=0.6):
    t = np.linspace(-length / 2, length / 2, 10)
    theta = np.linspace(0, 2 * np.pi, 14)
    T_g, Theta = np.meshgrid(t, theta)
    X_local = radius * np.cos(Theta)
    Y_local = radius * np.sin(Theta)
    Z_local = T_g
    pts_local = np.stack([X_local.ravel(), Y_local.ravel(), Z_local.ravel()], axis=1)
    pts_world = (rotation_matrix @ pts_local.T).T + center
    ax.plot_surface(
        pts_world[:, 0].reshape(X_local.shape),
        pts_world[:, 1].reshape(Y_local.shape),
        pts_world[:, 2].reshape(Z_local.shape),
        color=color, alpha=alpha, linewidth=0)


def draw_static_scene(ax, bounds, obstacles, start_pos, goal_pos):
    """绘制静态场景元素（边界、障碍物、起终点）"""
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

    for obs in obstacles:
        pos = obs['pos']
        if obs['type'] == 'box':
            plot_cube(ax, pos, obs['size'], color='#FF6B6B', alpha=0.4)
        elif obs['type'] == 'sphere':
            plot_sphere_surf(ax, pos, obs['radius'], color='#4ECDC4', alpha=0.4)
        elif obs['type'] == 'cylinder':
            plot_cylinder_surf(ax, pos, obs['radius'], obs['height'], np.eye(3), color='#FFE66D', alpha=0.4)

    ax.scatter(*start_pos, c='lime', s=120, marker='o', edgecolors='black', zorder=5, label='Start')
    ax.scatter(*goal_pos, c='red', s=120, marker='X', linewidths=2, zorder=5, label='Goal')


def draw_robot(ax, world_visuals, joints_data, link_transforms, robot_artists=None):
    """绘制机械臂并返回artist列表（用于动画清除）"""
    artists = []
    for vis in world_visuals:
        geom = vis['geometry']
        T = vis['transform']
        pos = T[:3, 3]
        R = T[:3, :3]
        gtype = geom.get('type')
        if gtype == 'sphere':
            r = float(geom['radius'])
            u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
            x = pos[0] + r * np.cos(u) * np.sin(v)
            y = pos[1] + r * np.sin(u) * np.sin(v)
            z = pos[2] + r * np.cos(v)
            s = ax.plot_surface(x, y, z, color='#FFA500', alpha=0.9, linewidth=0)
            artists.append(s)
        elif gtype == 'cylinder':
            r = float(geom['radius'])
            l = float(geom['length'])
            t = np.linspace(-l / 2, l / 2, 8)
            theta = np.linspace(0, 2 * np.pi, 10)
            T_g, Theta = np.meshgrid(t, theta)
            X_l = r * np.cos(Theta)
            Y_l = r * np.sin(Theta)
            Z_l = T_g
            pts_l = np.stack([X_l.ravel(), Y_l.ravel(), Z_l.ravel()], axis=1)
            pts_w = (R @ pts_l.T).T + pos
            s = ax.plot_surface(
                pts_w[:, 0].reshape(X_l.shape),
                pts_w[:, 1].reshape(Y_l.shape),
                pts_w[:, 2].reshape(Z_l.shape),
                color='#C0C0C0', alpha=0.8, linewidth=0)
            artists.append(s)

    # 骨架线
    jpositions = []
    for jnt in joints_data:
        child_T = link_transforms.get(jnt['child'])
        if child_T is not None:
            jpositions.append(child_T[:3, 3])
    if jpositions:
        jp = np.array(jpositions)
        line, = ax.plot(jp[:, 0], jp[:, 1], jp[:, 2], 'b-o', linewidth=2, markersize=4, alpha=0.8)
        artists.append(line)

    return artists


# ───────── 在路径点之间做线性插值 ─────────

def interpolate_path(path, n_interp=5):
    """在相邻路径点之间插入额外帧"""
    if len(path) < 2:
        return path
    smooth = [path[0]]
    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, n_interp + 1)[1:]:
            smooth.append(path[i] + t * (path[i + 1] - path[i]))
    return smooth


# ───────── 主函数 ─────────

def main(scene_id=0):
    # 1. 加载场景
    filename = os.path.join(DATA_DIR, f"env_{scene_id:05d}.json")
    with open(filename, 'r') as f:
        data = json.load(f)

    bounds = data['workspace_bounds']
    obstacles = data['obstacles']
    start_pos = np.array(data['start_pos'])
    goal_pos = np.array(data['goal_pos'])
    robot_base = data.get('robot_base_pos', [0.0, 0.0, 0.0])
    base_pos = tuple(robot_base)

    print(f"场景 {scene_id}: start={start_pos}, goal={goal_pos}, base={base_pos}")

    # 2. 解析 URDF
    links, joints = parse_urdf(URDF_PATH)
    revolute = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    n_dof = len(revolute)
    ee_link = revolute[-1]['child']

    print(f"机械臂 DOF={n_dof}, 末端link={ee_link}")

    # 3. IK 求解起终点
    print("正在求解起点 IK ...")
    q_start, ok_s = solve_ik_multi_start(
        links, joints, start_pos, base_pos=base_pos,
        joint_limits=JOINT_LIMITS, n_starts=30, tol=0.01, ee_link=ee_link
    )
    if not ok_s:
        # 即使没精确到达，也尝试继续
        ja = {revolute[k]['name']: q_start[k] for k in range(n_dof)}
        _, lt, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos)
        err = np.linalg.norm(lt[ee_link][:3, 3] - start_pos)
        print(f"  起点 IK 未精确收敛，误差={err:.4f}")
        if err > 0.05:
            print("  误差过大，退出"); return
    else:
        print(f"  起点 IK 成功")

    print("正在求解终点 IK ...")
    q_goal, ok_g = solve_ik_multi_start(
        links, joints, goal_pos, base_pos=base_pos,
        joint_limits=JOINT_LIMITS, n_starts=30, tol=0.01, ee_link=ee_link
    )
    if not ok_g:
        ja = {revolute[k]['name']: q_goal[k] for k in range(n_dof)}
        _, lt, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos)
        err = np.linalg.norm(lt[ee_link][:3, 3] - goal_pos)
        print(f"  终点 IK 未精确收敛，误差={err:.4f}")
        if err > 0.05:
            print("  误差过大，退出"); return
    else:
        print(f"  终点 IK 成功")

    # 4. RRT-Connect 规划
    print("正在进行 RRT-Connect 路径规划 ...")
    planner = RRTConnect(
        links, joints, obstacles, base_pos=base_pos,
        joint_limits=JOINT_LIMITS, step_size=0.2, max_iter=5000
    )
    path = planner.plan(q_start, q_goal)
    if path is None:
        print("路径规划失败"); return
    print(f"原始路径: {len(path)} 个节点")

    # 路径平滑
    path = shortcut_path(planner, path, n_attempts=50)
    print(f"平滑后路径: {len(path)} 个节点")

    # 帧间插值让动画更流畅
    frames = interpolate_path(path, n_interp=5)
    print(f"动画帧数: {len(frames)}")

    # 5. 可视化动画
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    draw_static_scene(ax, bounds, obstacles, start_pos, goal_pos)
    ax.scatter(*base_pos, c='blue', s=80, marker='s', edgecolors='black', zorder=5, label='Robot Base')

    # 绘制末端轨迹
    ee_traj = []
    for q in frames:
        ja = {revolute[k]['name']: q[k] for k in range(n_dof)}
        _, lt, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos)
        ee_traj.append(lt[ee_link][:3, 3].copy())
    ee_traj = np.array(ee_traj)
    ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], 'g--', linewidth=1.5, alpha=0.6, label='EE trajectory')

    # 坐标轴设置
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid = [(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=25, azim=-50)

    # 动画
    robot_artists = []
    title_text = ax.set_title('')

    def update(frame_idx):
        nonlocal robot_artists
        # 清除上一帧的机械臂
        for a in robot_artists:
            a.remove()
        robot_artists = []

        q = frames[frame_idx]
        ja = {revolute[k]['name']: q[k] for k in range(n_dof)}
        wv, lt, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos)
        robot_artists = draw_robot(ax, wv, joints, lt)

        ee_pos = lt[ee_link][:3, 3]
        title_text.set_text(
            f'Scene {scene_id:05d} | Frame {frame_idx+1}/{len(frames)} | '
            f'EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})'
        )
        return robot_artists

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=80, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sid = 0
    if len(sys.argv) > 1:
        sid = int(sys.argv[1])
    main(sid)
