import numpy as np 
from robot_utils.rotation_matrix import rotation_matrix_axis_angle, rpy_to_rotation, make_transform

def compute_link_transforms(joints, joint_angles=None, base_pos=(0, 0, 0), _cache=None):
    """
    轻量正运动学：只返回 link_transforms，不计算 visual/collision 几何。
    用于 IK 等高频调用场景。

    Args:
        _cache: 可选的预计算缓存 dict，包含 'joint_origin_T' 列表。
                通过 precompute_joint_cache(joints) 获取。
    """
    if joint_angles is None:
        joint_angles = {}

    link_transforms = {}
    T_base = np.eye(4)
    T_base[:3, 3] = np.array(base_pos)
    link_transforms['base'] = T_base

    if _cache is not None:
        for i, joint in enumerate(joints):
            parent_T = link_transforms.get(joint['parent'], np.eye(4))
            T_joint_origin = _cache['joint_origin_T'][i]
            if joint['type'] == 'revolute':
                angle = joint_angles.get(joint['name'], 0.0)
                R_joint = rotation_matrix_axis_angle(joint['axis'], angle)
                T_rot = np.eye(4)
                T_rot[:3, :3] = R_joint
                child_T = parent_T @ T_joint_origin @ T_rot
            else:
                child_T = parent_T @ T_joint_origin
            link_transforms[joint['child']] = child_T
    else:
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

    return link_transforms


def precompute_joint_cache(joints):
    """预计算关节静态变换矩阵，供 compute_link_transforms 高频调用时复用。"""
    return {
        'joint_origin_T': [make_transform(j['xyz'], j['rpy']) for j in joints]
    }


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


def precompute_ik_cache(joints):
    """预计算 IK 内循环所需的全部静态数据（关节原点变换、旋转轴反对称矩阵等）。"""
    origin_Ts = [make_transform(j['xyz'], j['rpy']) for j in joints]
    revolute_mask = []
    K_list = []
    KK_list = []
    axes = []
    for j in joints:
        is_rev = j['type'] in ('revolute', 'continuous')
        revolute_mask.append(is_rev)
        if is_rev:
            axis = np.array(j['axis'], dtype=float)
            n = np.linalg.norm(axis)
            if n > 1e-12:
                axis = axis / n
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            K_list.append(K)
            KK_list.append(K @ K)
            axes.append(axis)
    return {
        'origin_Ts': origin_Ts,
        'revolute_mask': revolute_mask,
        'K_list': K_list,
        'KK_list': KK_list,
        'axes': axes,
    }


def fk_for_ik(joints, q, base_T, ik_cache, ee_link):
    """
    IK 专用快速 FK：一次遍历同时计算 link 变换与各关节世界坐标系。

    Args:
        q: (n_dof,) 可动关节角度数组（按关节链顺序）
        base_T: 4x4 基座齐次变换
        ik_cache: precompute_ik_cache 返回的缓存
        ee_link: 末端 link 名称

    Returns:
        ee_pos: (3,) 末端位置
        joint_frames: 长度 n_dof 的列表，每个元素是 4x4 矩阵，
                      表示该关节旋转前的世界坐标系变换（用于解析雅可比）
    """
    origin_Ts = ik_cache['origin_Ts']
    rev_mask = ik_cache['revolute_mask']
    K_list = ik_cache['K_list']
    KK_list = ik_cache['KK_list']

    link_T = {'base': base_T}
    rev_idx = 0
    joint_frames = []

    for i, joint in enumerate(joints):
        parent_T = link_T.get(joint['parent'], base_T)
        if rev_mask[i]:
            T_joint = parent_T @ origin_Ts[i]
            joint_frames.append(T_joint)
            angle = q[rev_idx]
            s, c = np.sin(angle), np.cos(angle)
            T_rot = np.eye(4)
            T_rot[:3, :3] = np.eye(3) + s * K_list[rev_idx] + (1 - c) * KK_list[rev_idx]
            link_T[joint['child']] = T_joint @ T_rot
            rev_idx += 1
        else:
            link_T[joint['child']] = parent_T @ origin_Ts[i]

    return link_T[ee_link][:3, 3].copy(), joint_frames
