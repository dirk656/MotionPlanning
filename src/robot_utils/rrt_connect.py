import numpy as np
from robot_utils.compute_kinematic import compute_fk

class RRTConnect:
    """
    RRT-Connect 路径规划器
    用于在关节空间进行无碰撞路径规划
    """

    def __init__(self, q_start, q_goal, step_len=0.1, iter_max=5000,
                 links=None, joints=None, obstacles=None, clearance=0.05,
                 base_pos=(0.0, 0.0, 0.0), joint_limits=None, urdf_path=None):
        """
        初始化规划器
        :param q_start: 起始关节角度 (numpy array)
        :param q_goal: 目标关节角度 (numpy array)
        :param step_len: 扩展步长
        :param iter_max: 最大迭代次数
        :param links: URDF 解析后的 links 数据
        :param joints: URDF 解析后的 joints 数据
        :param obstacles: 障碍物列表
        :param clearance: 碰撞安全距离
        :param base_pos: 机器人基座位置
        :param joint_limits: 关节限位 [n, 2]
        """
        self.q_start = np.array(q_start)
        self.q_goal = np.array(q_goal)
        self.step_len = step_len
        self.iter_max = iter_max
        self.links = links
        self.joints = joints
        self.obstacles = obstacles
        self.clearance = clearance
        self.base_pos = base_pos
        self.joint_limits = joint_limits
        self.urdf_path = urdf_path
        
        # 维度
        self.n_dof = len(q_start)
        
        # 树结构 (使用列表动态存储，避免预分配浪费)
        # Tree A (Start Tree)
        self.vertices_a = [] 
        self.parent_a = []
        
        # Tree B (Goal Tree)
        self.vertices_b = []
        self.parent_b = []

    def planning(self):
        """
        执行 RRT-Connect 规划
        :return: 路径 (numpy array) 或 None (如果失败)
        """
        # 初始化两棵树
        self.vertices_a = [self.q_start.copy()]
        self.parent_a = [-1]  # -1 表示根节点

        self.vertices_b = [self.q_goal.copy()]
        self.parent_b = [-1]

        swapped = False
        for k in range(self.iter_max):
            q_rand = self._sample_random()

            # 扩展 A 树一步
            status_a, idx_a = self._extend_tree(self.vertices_a, self.parent_a, q_rand)

            # 只要没有被阻挡，就尝试让 B 树连接到 A 的新节点
            if status_a != 'Trapped':
                q_new_a = self.vertices_a[idx_a]
                status_b, idx_b = self._connect_tree(self.vertices_b, self.parent_b, q_new_a)

                if status_b == 'Reached':
                    path_a = self._extract_path(self.vertices_a, self.parent_a, idx_a)
                    path_b = self._extract_path(self.vertices_b, self.parent_b, idx_b)

                    # path_a: root_a -> connect, path_b: root_b -> connect
                    if swapped:
                        # A 是 goal 树，B 是 start 树
                        return np.vstack([path_b[:-1], path_a[::-1]])
                    # A 是 start 树，B 是 goal 树
                    return np.vstack([path_a[:-1], path_b[::-1]])

            # 双树轮换
            self.vertices_a, self.vertices_b = self.vertices_b, self.vertices_a
            self.parent_a, self.parent_b = self.parent_b, self.parent_a
            swapped = not swapped

            if k % 1000 == 0:
                print(f"  RRT Iteration: {k}, Nodes: {len(self.vertices_a) + len(self.vertices_b)}")

        return None

    def _sample_random(self):
        """随机采样关节空间"""
        # 95% 概率随机采样，5% 概率采样目标点（贪婪策略，加速收敛）
        if np.random.random() < 0.05:
            return self.q_goal.copy() # 注意：这里简化处理，实际应根据当前扩展方向决定 bias
        
        q_rand = np.random.uniform(
            self.joint_limits[:, 0], 
            self.joint_limits[:, 1]
        )
        return q_rand

    def _nearest_neighbor(self, q, vertices):
        """在 vertices 中找到距离 q 最近的节点索引"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, v in enumerate(vertices):
            dist = np.linalg.norm(q - v)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def _extend_tree(self, vertices, parents, q_target):
        """
        从给定树向 q_target 扩展一步。
        :return: (状态, 新/最近节点索引)
        """
        nearest_idx = self._nearest_neighbor(q_target, vertices)
        q_near = vertices[nearest_idx]
        direction = q_target - q_near
        dist = np.linalg.norm(direction)

        if dist == 0:
            return 'Reached', nearest_idx

        if dist > self.step_len:
            direction = (direction / dist) * self.step_len
            q_new = q_near + direction
            status = 'Advanced'
        else:
            q_new = q_target
            status = 'Reached'

        if self._is_edge_valid(q_near, q_new):
            vertices.append(q_new)
            parents.append(nearest_idx)
            return status, len(vertices) - 1

        return 'Trapped', nearest_idx

    def _connect_tree(self, vertices, parents, q_target):
        """
        连续扩展直到到达 q_target 或被阻挡。
        :return: (状态, 到达/停止节点索引)
        """
        while True:
            status, idx = self._extend_tree(vertices, parents, q_target)
            if status == 'Reached':
                return 'Reached', idx
            if status == 'Trapped':
                return 'Trapped', idx

    def _is_edge_valid(self, q_start, q_end):
        """
        检查从 q_start 到 q_end 的直线段是否无碰撞
        使用插值检测
        """
        dist = np.linalg.norm(q_end - q_start)
        if dist == 0:
            return True
            
        # 计算需要检查的点数
        # 确保每段插值不超过 step_len / 2，增加检测密度防止穿模
        n_checks = int(np.ceil(dist / (self.step_len / 2.0)))
        
        # 生成插值点
        for i in range(1, n_checks + 1):
            alpha = float(i) / n_checks
            q_check = (1 - alpha) * q_start + alpha * q_end
            
            # 检查关节限位
            if not self._check_joint_limits(q_check):
                return False
                
            # 检查碰撞
            if self.is_collision_free(q_check) is False:
                return False
                
        return True

    def _check_joint_limits(self, q):
        """检查关节角度是否在限位内"""
        return np.all((q >= self.joint_limits[:, 0] - 1e-6) & (q <= self.joint_limits[:, 1] + 1e-6))

    def is_collision_free(self, q):
        """
        检查给定构型 q 是否发生碰撞
        这是核心碰撞检测函数
        """
        # 1. 计算正运动学
        # 构造关节角度字典
        revolute = [j for j in self.joints if j['type'] in ('revolute', 'continuous')]
        joint_angles = {revolute[k]['name']: q[k] for k in range(self.n_dof)}
        
        _, link_transforms, _ = compute_fk(
            self.links,
            self.joints,
            joint_angles=joint_angles,
            base_pos=self.base_pos,
            urdf_path=self.urdf_path,
        )
        
        # 2. 检查机器人自身与障碍物的碰撞
        robot_points = self._sample_robot_points(link_transforms)
        if not self.obstacles:
            return True

        for obs in self.obstacles:
            obs_type = obs.get('type')
            obs_pos_raw = obs.get('pos', obs.get('position'))
            if obs_pos_raw is None:
                continue
            obs_pos = np.array(obs_pos_raw, dtype=float)

            for r_pt in robot_points:
                dist = float('inf')
                if obs_type == 'sphere':
                    radius = float(obs.get('radius', 0.0))
                    dist = np.linalg.norm(r_pt - obs_pos) - radius
                elif obs_type == 'cylinder':
                    radius = float(obs.get('radius', 0.0))
                    height = float(obs.get('height', 0.0))
                    # 圆柱体碰撞检测简化版 (投影到XY平面 + Z轴高度)
                    dz = abs(r_pt[2] - obs_pos[2])
                    if dz > height / 2.0: # 高度超出
                         dist = np.linalg.norm(r_pt[:2] - obs_pos[:2]) # 只看平面距离
                    else:
                        dist = np.linalg.norm(r_pt[:2] - obs_pos[:2]) - radius
                elif obs_type == 'box':
                    size = np.array(obs.get('size', [0.0, 0.0, 0.0]), dtype=float)
                    half = size / 2.0
                    inside = np.all(np.abs(r_pt - obs_pos) <= (half + self.clearance))
                    if inside:
                        return False
                
                if dist < self.clearance:
                    return False # 发生碰撞
        return True

    def _sample_robot_points(self, link_transforms):
        """
        从机器人连杆中采样关键点用于碰撞检测
        """
        points = []
        # 这里简化处理，只采样每个连杆变换矩阵的原点
        # 实际应用中应该采样连杆的包围盒顶点
        if isinstance(self.links, dict):
            for link_name in self.links.keys():
                if link_name in link_transforms:
                    pos = link_transforms[link_name][:3, 3]
                    points.append(pos)
        else:
            for link in self.links:
                link_name = link.get('name') if isinstance(link, dict) else None
                if link_name is not None and link_name in link_transforms:
                    pos = link_transforms[link_name][:3, 3]
                    points.append(pos)
        return points

    def _extract_path(self, vertices, parents, idx):
        """
        从树中提取路径
        :param vertices: 节点列表
        :param parents: 父节点索引列表
        :param idx: 当前节点索引
        :return: 路径列表
        """
        path = []
        curr_idx = idx
        while curr_idx != -1:
            path.append(vertices[curr_idx])
            curr_idx = parents[curr_idx]
        path.reverse()
        return np.array(path)