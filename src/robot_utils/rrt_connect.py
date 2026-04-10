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

    def __init__(
        self,
        q_start,
        q_goal,
        step_len,
        iter_max,
        links,
        joints,
        obstacles,
        clearance=0.01,
        base_pos=(0, 0, 0),
        joint_limits=None,
        collision_samples_per_link=8,
    ):
        self.q_start = np.array(q_start).astype(np.float64)
        self.q_goal = np.array(q_goal).astype(np.float64)
        self.step_len = step_len
        self.iter_max = iter_max
        self.clearance = clearance

        self.links = links
        self.joints = joints
        self.base_pos = base_pos
        self.n_samples = collision_samples_per_link

        self.revolute_joints = [j for j in joints if j['type'] in ('revolute', 'continuous')]
        self.n_dof = len(self.revolute_joints)
        self.joint_names = [j['name'] for j in self.revolute_joints]

        if joint_limits is not None:
            self.joint_limits = np.array(joint_limits)
        else:
            self.joint_limits = np.array([[-np.pi, np.pi]] * self.n_dof)

        # 预分配双树节点数组
        self.vertices_a = np.zeros((1 + iter_max, self.n_dof))
        self.vertex_parents_a = np.zeros(1 + iter_max).astype(int)
        self.vertices_a[0] = self.q_start
        self.num_vertices_a = 1

        self.vertices_b = np.zeros((1 + iter_max, self.n_dof))
        self.vertex_parents_b = np.zeros(1 + iter_max).astype(int)
        self.vertices_b[0] = self.q_goal
        self.num_vertices_b = 1

        self.path = []

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

    def planning(self):
        """
        RRT-Connect 双树规划。
        - outputs:
            - path: np.array (n_path, n_dof)，或空列表表示规划失败
        """
        if not self.is_collision_free(self.q_start):
            print("警告: 起始配置存在碰撞")
            return []
        if not self.is_collision_free(self.q_goal):
            print("警告: 目标配置存在碰撞")
            return []

        swapped = False
        for k in range(self.iter_max):
            q_rand = self.sample_random()

            # 扩展 tree_a
            status_a, idx_a, self.num_vertices_a = self._extend_tree(
                self.vertices_a, self.vertex_parents_a, self.num_vertices_a, q_rand
            )
            if status_a != 'trapped':
                q_new_a = self.vertices_a[idx_a]
                # 从 tree_b 连接到 tree_a 的新节点
                status_b, idx_b, self.num_vertices_b = self._connect_tree(
                    self.vertices_b, self.vertex_parents_b, self.num_vertices_b, q_new_a
                )
                if status_b == 'reached':
                    path_a = self._extract_tree_path(
                        self.vertices_a, self.vertex_parents_a, idx_a
                    )
                    path_b = self._extract_tree_path(
                        self.vertices_b, self.vertex_parents_b, idx_b
                    )
                    # tree_a 路径: root_a → connection
                    # tree_b 路径: root_b → connection, 需要反转
                    if swapped:
                        # tree_a=goal树, tree_b=start树
                        self.path = np.vstack([path_b[:-1], path_a[::-1]])
                    else:
                        # tree_a=start树, tree_b=goal树
                        self.path = np.vstack([path_a[:-1], path_b[::-1]])
                    return self.path

            # 交换两棵树以均衡生长
            self.vertices_a, self.vertices_b = self.vertices_b, self.vertices_a
            self.vertex_parents_a, self.vertex_parents_b = self.vertex_parents_b, self.vertex_parents_a
            self.num_vertices_a, self.num_vertices_b = self.num_vertices_b, self.num_vertices_a
            swapped = not swapped

            if (k + 1) % 1000 == 0:
                print(k + 1)

        print(f"RRT-Connect: {self.iter_max} 次迭代后未找到路径")
        return []

    def _extend_tree(self, vertices, vertex_parents, num_vertices, q_target):
        """
        向 q_target 方向扩展一步。
        - outputs:
            - status: 'reached' | 'advanced' | 'trapped'
            - idx: 新/最近节点索引
            - num_vertices: 更新后的节点数
        """
        node_nearest, nearest_index = self.nearest_neighbor(
            vertices[:num_vertices], q_target
        )
        q_new = self.steer(node_nearest, q_target)

        if self.is_collision_free(q_new) and self.is_edge_valid(node_nearest, q_new):
            if np.linalg.norm(q_new - node_nearest) < 1e-8:
                # 新节点与最近节点重合，不添加
                if np.linalg.norm(q_new - q_target) < 1e-6:
                    return 'reached', nearest_index, num_vertices
                return 'advanced', nearest_index, num_vertices

            new_index = num_vertices
            vertices[new_index] = q_new
            vertex_parents[new_index] = nearest_index
            num_vertices += 1

            if np.linalg.norm(q_new - q_target) < 1e-6:
                return 'reached', new_index, num_vertices
            return 'advanced', new_index, num_vertices

        return 'trapped', -1, num_vertices

    def _connect_tree(self, vertices, vertex_parents, num_vertices, q_target):
        """
        反复扩展直到到达 q_target 或被阻挡。
        - outputs:
            - status: 'reached' | 'trapped'
            - idx: 到达/停止节点索引
            - num_vertices: 更新后的节点数
        """
        while True:
            status, idx, num_vertices = self._extend_tree(
                vertices, vertex_parents, num_vertices, q_target
            )
            if status == 'reached':
                return 'reached', idx, num_vertices
            if status == 'trapped':
                return 'trapped', idx, num_vertices

    def _extract_tree_path(self, vertices, vertex_parents, end_index):
        """
        从 end_index 回溯到根节点 (index 0)，提取路径。
        - outputs:
            - path: np.array (n_path, n_dof)
        """
        path = [vertices[end_index].copy()]
        idx = end_index
        while idx != 0:
            idx = vertex_parents[idx]
            path.append(vertices[idx].copy())
        path.reverse()
        return np.stack(path, axis=0)

    def steer(self, q_near, q_target):
        """
        从 q_near 向 q_target 方向步进，步幅不超过 step_len。
        """
        dist, direction = self.get_distance_and_direction(q_near, q_target)
        dist = min(self.step_len, dist)
        return q_near + dist * direction

    def sample_random(self):
        lo = self.joint_limits[:, 0]
        hi = self.joint_limits[:, 1]
        return lo + np.random.rand(self.n_dof) * (hi - lo)

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
            if np.any(points_in_AABB_3d(arm_pts, self.box_arr, clearance=self.clearance)):
                return False
        if self.sphere_arr is not None:
            if np.any(points_in_ball_3d(arm_pts, self.sphere_arr, clearance=self.clearance)):
                return False
        if self.cylinder_arr is not None:
            if np.any(points_in_cylinder_3d(arm_pts, self.cylinder_arr, clearance=self.clearance)):
                return False
        return True

    def is_edge_valid(self, q1, q2, n_checks=None):
        """检查 q1 到 q2 之间的线性插值路径是否无碰撞"""
        if n_checks is None:
            dist = np.linalg.norm(np.array(q1) - np.array(q2))
            n_checks = max(5, int(dist / (self.step_len / 2.0)))
        for t in np.linspace(0, 1, n_checks):
            q_mid = q1 + t * (q2 - q1)
            if not self.is_collision_free(q_mid):
                return False
        return True

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

    def check_success(self, path):
        if path is None or len(path) == 0:
            return False
        return np.allclose(path[0], self.q_start) and np.allclose(path[-1], self.q_goal)

    def get_path_len(self, path):
        if path is None or len(path) == 0:
            return np.inf
        path = np.array(path)
        path_disp = path[1:] - path[:-1]
        return np.linalg.norm(path_disp, axis=1).sum()

    @staticmethod
    def nearest_neighbor(node_list, n):
        '''
        find the node in node_list which is the closest to n.
        - inputs:
            - node_list: np (num_vertices, n_dof)
            - n: np (n_dof,)
        - outputs:
            - nearest_n: np (n_dof,)
            - nearest_index
        '''
        vec_to_n = n - node_list
        nearest_index = np.argmin(np.linalg.norm(vec_to_n, axis=1))
        return node_list[nearest_index], nearest_index

    @staticmethod
    def get_distance_and_direction(node_start, node_end):
        '''
        - inputs:
            - node_start, node_end: np (n_dof,)
        - outputs:
            - distance
            - direction unit vector, np (n_dof,)
        '''
        diff = node_end - node_start
        distance = np.linalg.norm(diff)
        if distance == 0:
            return 0, np.zeros_like(diff)
        direction = diff / distance
        return distance, direction


import numpy as np

def shortcut_path(planner, path, n_attempts=100, max_failures=20):
    """
    增强版路径平滑：
    1. 优先尝试长距离跨越 (更容易大幅缩短路径)。
    2. 引入连续失败计数，若连续多次无法找到捷径则提前终止。
    3. 动态调整采样范围。
    """
    if path is None or len(path) < 3:
        return path
    
    path = list(path)
    consecutive_failures = 0
    
    # 当路径还能被优化且未达到最大尝试次数时循环
    attempt = 0
    while attempt < n_attempts and consecutive_failures < max_failures and len(path) >= 3:
        attempt += 1
        
        # 策略：偏向于选择跨度较大的 i, j (例如至少跨越当前路径长度的 10% 或 2个点)
        min_span = max(2, int(len(path) * 0.1)) 
        if len(path) <= min_span + 1:
            break
            
        i = np.random.randint(0, len(path) - min_span - 1)
        # j 至少比 i 大 min_span
        j = np.random.randint(i + min_span, len(path))
        
        # 碰撞检测
        if planner.is_edge_valid(path[i], path[j]):
            # 成功捷径：截断中间部分
            # 注意：path[:i+1] 包含 i, path[j:] 包含 j
            new_path = path[:i + 1] + path[j:]
            
            # 可选：如果新路径太短，直接返回
            if len(new_path) < 3:
                return np.array(new_path)
                
            path = new_path
            consecutive_failures = 0 # 重置失败计数
            # 成功一次后，稍微减少剩余尝试次数以节省时间，或者继续
        else:
            consecutive_failures += 1
            
    return np.array(path)