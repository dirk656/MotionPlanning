import numpy as np 
from robot_utils.rotation_matrix import rotation_matrix_axis_angle, rpy_to_rotation, make_transform

def compute_fk(links, joints, joint_angles=None, base_pos=(0, 0, 0)):
    """
    计算正运动学。

    Args:
        links: parse_urdf 返回的 links 字典
        joints: parse_urdf 返回的 joints 列表
        joint_angles: dict {joint_name: angle_rad}，默认全零
        base_pos: 机械臂基座在世界坐标系中的位置

    Returns:
        world_visuals: 世界坐标系下的 visual 几何列表
            [{'link', 'transform' (4x4), 'geometry'}, ...]
        link_transforms: dict {link_name: 4x4 np.array}
        world_collisions: 世界坐标系下的 collision 几何列表
            [{'link', 'transform' (4x4), 'geometry'}, ...]
    """
    if joint_angles is None:
        joint_angles = {}

    link_transforms = {}
    T_base = np.eye(4)
    T_base[:3, 3] = np.array(base_pos)
    link_transforms['base'] = T_base

    for joint in joints:
        parent_T = link_transforms.get(joint['parent'], np.eye(4))
        T_joint_origin = make_transform(joint['xyz'], joint['rpy'])

        if joint['type'] == 'revolute':
            angle = joint_angles.get(joint['name'], 0.0)
            R_joint = rotation_matrix_axis_angle(joint['axis'], angle)
            T_rot = np.eye(4)
            T_rot[:3, :3] = R_joint
            child_T = parent_T @ T_joint_origin @ T_rot
        else:
            child_T = parent_T @ T_joint_origin

        link_transforms[joint['child']] = child_T

    world_visuals = []
    for link_name, link_info in links.items():
        T_link = link_transforms.get(link_name, np.eye(4))
        for vis in link_info['visuals']:
            T_vis = make_transform(vis['xyz'], vis['rpy'])
            T_world = T_link @ T_vis
            world_visuals.append({
                'link': link_name,
                'transform': T_world,
                'geometry': vis['geometry'],
            })

    world_collisions = []
    for link_name, link_info in links.items():
        T_link = link_transforms.get(link_name, np.eye(4))
        for col in link_info.get('collisions', []):
            T_col = make_transform(col['xyz'], col['rpy'])
            T_world = T_link @ T_col
            world_collisions.append({
                'link': link_name,
                'transform': T_world,
                'geometry': col['geometry'],
            })

    return world_visuals, link_transforms, world_collisions
