
import xml.etree.ElementTree as ET
import numpy as np

from robot_utils.rotation_matrix import * 


def parse_urdf(urdf_path):
    """
    解析 URDF 文件，返回 links 和 joints 信息。
    """

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link_elem in root.findall('link'):
        name = link_elem.get('name')
        visuals = []
        for vis in link_elem.findall('visual'):
            origin = vis.find('origin')

            xyz = parse_vec(origin.get('xyz') if origin is not None else None)
            rpy = parse_vec(origin.get('rpy') if origin is not None else None)

            geom = vis.find('geometry')
            geom_info = {}

            for child in geom:
                geom_info['type'] = child.tag
                for k, v in child.attrib.items():
                    geom_info[k] = v

            visuals.append({'xyz': xyz, 'rpy': rpy, 'geometry': geom_info})#作为字典加入列表

            
        collisions = []
        for col in link_elem.findall('collision'):
            origin = col.find('origin')
            xyz = parse_vec(origin.get('xyz') if origin is not None else None)
            rpy = parse_vec(origin.get('rpy') if origin is not None else None)
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
        xyz = parse_vec(origin.get('xyz') if origin is not None else None)
        rpy = parse_vec(origin.get('rpy') if origin is not None else None)
        axis_elem = joint_elem.find('axis')
        axis = parse_vec(axis_elem.get('xyz') if axis_elem is not None else None, (0, 0, 1))
        joints.append({
            'name': name, 'type': jtype,
            'parent': parent, 'child': child,
            'xyz': xyz, 'rpy': rpy, 'axis': axis,
        })
    

    # links are dict and joints are list 
    # links contain visuals and collisions info 
    # joints contain kinematic chain info
    return links, joints


