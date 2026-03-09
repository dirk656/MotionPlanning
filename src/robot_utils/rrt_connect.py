import numpy as np
from robot_utils.fk_solver import compute_fk
from robot_utils.urdf_to_geometry import get_robot_collision_bodies
from planning_utils.collision_check_utils import (
    points_in_AABB_3d, points_in_ball_3d, points_in_cylinder_3d
)


class RRTConnect:
    """
    RRT-Connect 关节空间路径规划器。

    在关节空间中双向生长 RRT，通过 FK + 碰撞检测确保机械臂不与环境障碍物碰撞。
    """

    def __init__(self, links, joints, obstacles, base_pos=(0, 0, 0),
                 joint_limits=None, step_size=0.15, max_iter=5000,
                 collision_samples_per_link=8):
        self.links = links
        self.joints = joints
        self.base_pos = base_pos
        self.step_size = step_size
        self.max_iter = max_iter
        self.n_samples = collision_samples_per_link

        self.revolute_joints = [j for j in joints if j['type'] in ('revolute', 'continuous')]
        self.n_dof = len(self.revolute_joints)
        self.joint_names = [j['name'] for j in self.revolute_joints]

        if joint_limits is not None:
            self.joint_limits = np.array(joint_limits)
        else:
            self.joint_limits = np.array([[-np.pi, np.pi]] * self.n_dof)

        # 预处理环境障碍物
        boxes, spheres, cylinders = [], [], []
        for obs in obstacles:
            pos = np.array(obs['pos'])
            if obs['type'] == 'box':
                size = np.array(obs['size'])
                min_corner = pos - size / 2
                boxes.append([*min_corner, *size])
            elif obs['type'] == 'sphere':
                spheres.append([*pos, obs['radius']])
            elif obs['type'] == 'cylinder':
                cylinders.append([*pos, obs['radius'], obs['height']])
        self.box_arr = np.array(boxes) if boxes else None
        self.sphere_arr = np.array(spheres) if spheres else None
        self.cylinder_arr = np.array(cylinders) if cylinders else None

    def _q_to_joint_angles(self, q):
        return {self.joint_names[i]: q[i] for i in range(self.n_dof)}

    def _sample_arm_points(self, world_collisions):
        """从碰撞体中采样表面点用于碰撞检测"""
        bodies = get_robot_collision_bodies(world_collisions)
        pts = []
        for body in bodies:
            T = body['transform']
            pos = T[:3, 3]
            R = T[:3, :3]
            if body['type'] == 'sphere':
                r = body['radius']
                pts.append(pos)
                for ax in np.eye(3):
                    pts.append(pos + R @ (ax * r))
                    pts.append(pos - R @ (ax * r))
            elif body['type'] == 'cylinder':
                r = body['radius']
                h = body['length']
                for z in np.linspace(-h / 2, h / 2, self.n_samples):
                    pts.append(pos + R @ np.array([0, 0, z]))
                for z in [-h / 2, h / 2]:
                    for theta in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                        pts.append(pos + R @ np.array([r * np.cos(theta), r * np.sin(theta), z]))
        if not pts:
            return np.empty((0, 3))
        return np.vstack(pts)

    def is_collision_free(self, q):
        """检查关节配置 q 下机械臂是否与环境无碰撞"""
        q = np.array(q)
        if np.any(q < self.joint_limits[:, 0]) or np.any(q > self.joint_limits[:, 1]):
            return False

        ja = self._q_to_joint_angles(q)
        _, _, world_collisions = compute_fk(
            self.links, self.joints, joint_angles=ja, base_pos=self.base_pos
        )
        arm_pts = self._sample_arm_points(world_collisions)
        if len(arm_pts) == 0:
            return True

        if self.box_arr is not None:
            if np.any(points_in_AABB_3d(arm_pts, self.box_arr, clearance=0.01)):
                return False
        if self.sphere_arr is not None:
            if np.any(points_in_ball_3d(arm_pts, self.sphere_arr, clearance=0.01)):
                return False
        if self.cylinder_arr is not None:
            if np.any(points_in_cylinder_3d(arm_pts, self.cylinder_arr, clearance=0.01)):
                return False
        return True

    def _is_edge_valid(self, q1, q2, n_checks=10):
        """检查 q1 到 q2 之间的线性插值路径是否无碰撞"""
        for t in np.linspace(0, 1, n_checks):
            q_mid = q1 + t * (q2 - q1)
            if not self.is_collision_free(q_mid):
                return False
        return True

    def _sample_random(self):
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        return lo + np.random.rand(self.n_dof) * (hi - lo)

    def _nearest(self, tree, q):
        dists = [np.linalg.norm(np.array(node) - q) for node in tree]
        return int(np.argmin(dists))

    def _steer(self, q_near, q_rand):
        diff = q_rand - q_near
        dist = np.linalg.norm(diff)
        if dist <= self.step_size:
            return q_rand.copy()
        return q_near + (diff / dist) * self.step_size

    def _extend(self, tree, parents, q):
        idx_near = self._nearest(tree, q)
        q_near = tree[idx_near]
        q_new = self._steer(q_near, q)
        if self.is_collision_free(q_new) and self._is_edge_valid(q_near, q_new, 5):
            tree.append(q_new)
            parents.append(idx_near)
            if np.linalg.norm(q_new - q) < 1e-6:
                return 'reached', len(tree) - 1
            return 'advanced', len(tree) - 1
        return 'trapped', -1

    def _connect(self, tree, parents, q):
        while True:
            status, idx = self._extend(tree, parents, q)
            if status == 'reached':
                return 'reached', idx
            if status == 'trapped':
                return 'trapped', idx
        # 'advanced' 继续循环

    def _extract_path(self, tree, parents, idx):
        path = []
        i = idx
        while i != -1:
            path.append(tree[i])
            i = parents[i]
        path.reverse()
        return path

    def plan(self, q_start, q_goal):
        """
        RRT-Connect 规划。

        Args:
            q_start: (n_dof,) 起始关节角
            q_goal: (n_dof,) 目标关节角

        Returns:
            path: list of np.array (n_dof,)，或 None 表示规划失败
        """
        q_start = np.array(q_start, dtype=float)
        q_goal = np.array(q_goal, dtype=float)

        if not self.is_collision_free(q_start):
            print("警告: 起始配置存在碰撞")
            return None
        if not self.is_collision_free(q_goal):
            print("警告: 目标配置存在碰撞")
            return None

        tree_a = [q_start]
        parents_a = [-1]
        tree_b = [q_goal]
        parents_b = [-1]

        for i in range(self.max_iter):
            q_rand = self._sample_random()

            # 扩展 tree_a
            status_a, idx_a = self._extend(tree_a, parents_a, q_rand)
            if status_a != 'trapped':
                q_new_a = tree_a[idx_a]
                # 从 tree_b 连接到 tree_a 的新节点
                status_b, idx_b = self._connect(tree_b, parents_b, q_new_a)
                if status_b == 'reached':
                    path_a = self._extract_path(tree_a, parents_a, idx_a)
                    path_b = self._extract_path(tree_b, parents_b, idx_b)
                    path_b.reverse()
                    return path_a + path_b

            # 交换两棵树
            tree_a, tree_b = tree_b, tree_a
            parents_a, parents_b = parents_b, parents_a

        print(f"RRT-Connect: {self.max_iter} 次迭代后未找到路径")
        return None


def shortcut_path(planner, path, n_attempts=100):
    """路径平滑：随机选取两个路径点，若直线连接无碰撞则截断中间段"""
    if path is None or len(path) < 3:
        return path
    path = [np.array(q) for q in path]
    for _ in range(n_attempts):
        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))
        if planner._is_edge_valid(path[i], path[j], n_checks=15):
            path = path[:i + 1] + path[j:]
        if len(path) < 3:
            break
    return path
