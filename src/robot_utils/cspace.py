import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_utils import parse_urdf, compute_fk, get_robot_collision_bodies
from planning_utils.collision_check_utils import points_in_AABB_3d, points_in_ball_3d

class ManipulatorCSpaceEnv:
    def __init__(self, urdf_path, env_boxes, env_balls, joint_limits, base_pos=(0,0,0), n_samples_per_link=8):
        self.links, self.joints = parse_urdf(urdf_path)
        self.env_boxes = np.array(env_boxes) if env_boxes else np.empty((0, 6))
        self.env_balls = np.array(env_balls) if env_balls else np.empty((0, 4))
        self.joint_limits = np.array(joint_limits)
        self.base_pos = base_pos
        self.dof = len(joint_limits)
        self.n_samples = n_samples_per_link

    def _sample_points_from_bodies(self, world_collisions):
        bodies = get_robot_collision_bodies(world_collisions)
        all_points = []
        
        for body in bodies:
            T = body['transform']
            pos = T[:3, 3]
            R = T[:3, :3]
            
            if body['type'] == 'sphere':
                r = body['radius']
                all_points.append(pos)
                offsets = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]) * r
                for off in offsets:
                    all_points.append(pos + R @ off)
                    
            elif body['type'] == 'cylinder':
                r = body['radius']
                h = body['length']
                z_steps = np.linspace(-h/2, h/2, self.n_samples)
                for z in z_steps:
                    all_points.append(pos + R @ np.array([0, 0, z]))
                for z in [-h/2, h/2]:
                    for theta in np.linspace(0, 2*np.pi, 4, endpoint=False):
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        all_points.append(pos + R @ np.array([x, y, z]))
                        
        if not all_points:
            return np.empty((0, 3))
        return np.vstack(all_points)

    def is_valid_config(self, q):
        q = np.array(q)
        if np.any(q < self.joint_limits[:, 0]) or np.any(q > self.joint_limits[:, 1]):
            return False
            
        joint_angles = {}
        joint_idx = 0
        for joint in self.joints:
            if joint['type'] in ['revolute', 'continuous']:
                if joint_idx < len(q):
                    joint_angles[joint['name']] = q[joint_idx]
                    joint_idx += 1
        
        try:
            _, _, world_collisions = compute_fk(
                self.links, self.joints, joint_angles=joint_angles, base_pos=self.base_pos
            )
        except Exception:
            return False

        arm_points = self._sample_points_from_bodies(world_collisions)
        if len(arm_points) == 0:
            return True
            
        in_boxes = points_in_AABB_3d(arm_points, self.env_boxes.tolist(), clearance=0.0)
        in_balls = points_in_ball_3d(arm_points, self.env_balls.tolist(), clearance=0.0)
        
        return not (np.any(in_boxes) or np.any(in_balls))

    def sample_random_config(self):
        ranges = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        return self.joint_limits[:, 0] + np.random.rand(self.dof) * ranges

    def distance(self, q1, q2):
        return np.linalg.norm(np.array(q1) - np.array(q2))