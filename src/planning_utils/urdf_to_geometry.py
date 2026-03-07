"""
URDF 解析与正运动学计算工具。
从 URDF 文件中提取机械臂的运动链和 visual 几何信息，
计算给定关节角度下各 link 在世界坐标系中的位姿。
"""
import xml.etree.ElementTree as ET
import numpy as np


def rotation_matrix_axis_angle(axis, angle):
    """绕任意轴的旋转矩阵（Rodrigues 公式）"""
    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rpy_to_rotation(rpy):
    """RPY (roll, pitch, yaw) → 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    r, p, y = rpy
    Rx = rotation_matrix_axis_angle([1, 0, 0], r)
    Ry = rotation_matrix_axis_angle([0, 1, 0], p)
    Rz = rotation_matrix_axis_angle([0, 0, 1], y)
    return Rz @ Ry @ Rx


def make_transform(xyz, rpy):
    """构造 4x4 齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = rpy_to_rotation(rpy)
    T[:3, 3] = xyz
    return T


def _parse_vec(s, default=(0, 0, 0)):
    if s is None:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in s.split()], dtype=float)


def parse_urdf(urdf_path):
    """
    解析 URDF 文件，返回 links 和 joints 信息。

    Returns:
        links: dict {link_name: {'visuals': [{'xyz', 'rpy', 'geometry'}, ...]}}
        joints: list of {'name', 'type', 'parent', 'child', 'xyz', 'rpy', 'axis'}
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link_elem in root.findall('link'):
        name = link_elem.get('name')
        visuals = []
        for vis in link_elem.findall('visual'):
            origin = vis.find('origin')
            xyz = _parse_vec(origin.get('xyz') if origin is not None else None)
            rpy = _parse_vec(origin.get('rpy') if origin is not None else None)
            geom = vis.find('geometry')
            geom_info = {}
            for child in geom:
                geom_info['type'] = child.tag
                for k, v in child.attrib.items():
                    geom_info[k] = v
            visuals.append({'xyz': xyz, 'rpy': rpy, 'geometry': geom_info})
        collisions = []
        for col in link_elem.findall('collision'):
            origin = col.find('origin')
            xyz = _parse_vec(origin.get('xyz') if origin is not None else None)
            rpy = _parse_vec(origin.get('rpy') if origin is not None else None)
            geom = col.find('geometry')
            geom_info = {}
            for child in geom:
                geom_info['type'] = child.tag
                for k, v in child.attrib.items():
                    geom_info[k] = v
            collisions.append({'xyz': xyz, 'rpy': rpy, 'geometry': geom_info})
        links[name] = {'visuals': visuals, 'collisions': collisions}

    joints = []
    for joint_elem in root.findall('joint'):
        name = joint_elem.get('name')
        jtype = joint_elem.get('type')
        parent = joint_elem.find('parent').get('link')
        child = joint_elem.find('child').get('link')
        origin = joint_elem.find('origin')
        xyz = _parse_vec(origin.get('xyz') if origin is not None else None)
        rpy = _parse_vec(origin.get('rpy') if origin is not None else None)
        axis_elem = joint_elem.find('axis')
        axis = _parse_vec(axis_elem.get('xyz') if axis_elem is not None else None, (0, 0, 1))
        joints.append({
            'name': name, 'type': jtype,
            'parent': parent, 'child': child,
            'xyz': xyz, 'rpy': rpy, 'axis': axis,
        })

    return links, joints


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


def get_robot_collision_bodies(world_collisions):
    """
    将 compute_fk 返回的 world_collisions 转换为 points_in_robot_arm_3d 所需的格式。

    Returns:
        bodies: list of {'type', 'transform', 'radius', 'length'(cylinder only)}
    """
    bodies = []
    for col in world_collisions:
        geom = col['geometry']
        gtype = geom.get('type')
        if gtype == 'sphere':
            bodies.append({
                'type': 'sphere',
                'transform': col['transform'],
                'radius': float(geom['radius']),
            })
        elif gtype == 'cylinder':
            bodies.append({
                'type': 'cylinder',
                'transform': col['transform'],
                'radius': float(geom['radius']),
                'length': float(geom['length']),
            })
    return bodies
