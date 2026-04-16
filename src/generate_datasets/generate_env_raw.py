
import json
import os
import sys
import glob
import argparse
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "env")
URDF_PATH = os.path.join(_PROJECT_ROOT, "src", "robots", "franka_panda_gem.urdf")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot_utils.urdf_to_geometry import parse_urdf
from robot_utils.compute_kinematic import compute_fk
from robot_utils.compute_kinematic import compute_ik
from robot_utils.compute_kinematic import get_pinocchio_instance
from robot_utils.rrt_connect import RRTConnect


JOINT_LIMITS = np.array([
    [-2.9007, 2.9007],
    [-1.8361, 1.8361],
    [-2.9007, 2.9007],
    [-3.0770, -0.1169],
    [-2.8763, 2.8763],
    [0.4398, 4.6216],
    [-3.0508, 3.0508],
])

#get ee link name from urdf joints
def get_ee_link(joints):

    revolute = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    ee_link = revolute[-1]['child']
    while True:
        child_joint = next((j for j in joints if j['parent'] == ee_link and j['type'] == 'fixed'), None)
        if child_joint is None:
            break
        ee_link = child_joint['child']


    return ee_link


#数据集清洗
def validate_scene(links, joints, ee_link, scene_path, urdf_path):
   
    with open(scene_path, 'r') as f:
        data = json.load(f)
    #load data 

    obstacles = data['obstacles']
    start_pos = np.array(data['start_pos'])
    goal_pos = np.array(data['goal_pos'])

    #regulate base pos 
    base_pos = tuple(data.get('robot_base_pos', [0.0, 0.0, 0.0]))

    #load arm data 
    revolute = [j for j in joints if j['type'] in ('revolute', 'continuous')]
    n_dof = len(revolute)


    # 构建碰撞检查器
    col_checker = RRTConnect(
        np.zeros(n_dof), np.zeros(n_dof),
        step_len=0.1, iter_max=1,
        links=links, joints=joints, obstacles=obstacles,
        clearance=0.05, base_pos=base_pos, joint_limits=JOINT_LIMITS,
        urdf_path=urdf_path,
    )

    # IK 求解起点
    q_start, ok_s = compute_ik(
        links, joints, start_pos, base_pos=base_pos,
        joint_limits=JOINT_LIMITS, n_starts=50, tol=0.01, ee_link=ee_link,
        collision_checker=col_checker.is_collision_free,
        urdf_path=urdf_path,
    )
    if not ok_s:
        if q_start is None:
            return False
        ja = {revolute[k]['name']: q_start[k] for k in range(n_dof)}
        _, lt, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos, urdf_path=urdf_path)
        err = np.linalg.norm(lt[ee_link][:3, 3] - start_pos)
        if err > 0.05:
            return False

    # IK 求解终点
    q_goal, ok_g = compute_ik(
        links, joints, goal_pos, base_pos=base_pos,
        joint_limits=JOINT_LIMITS, n_starts=50, tol=0.01, ee_link=ee_link,
        collision_checker=col_checker.is_collision_free,
        urdf_path=urdf_path,
    )
    if not ok_g:
        if q_goal is None:
            return False
        ja = {revolute[k]['name']: q_goal[k] for k in range(n_dof)}
        _, lt, _ = compute_fk(links, joints, joint_angles=ja, base_pos=base_pos, urdf_path=urdf_path)
        err = np.linalg.norm(lt[ee_link][:3, 3] - goal_pos)
        if err > 0.05:
            return False

    # RRT-Connect 规划
    planner = RRTConnect(
        q_start, q_goal,
        step_len=0.1, iter_max=5000,
        links=links, joints=joints, obstacles=obstacles,
        clearance=0.05, base_pos=base_pos, joint_limits=JOINT_LIMITS,
        urdf_path=urdf_path,
    )
    path = planner.planning()
    if path is None or len(path) == 0:
        return False

    # 规划成功后，保存末端轨迹到场景文件
    ee_path = []
    
    for q in path:
        joint_angles = {revolute[k]['name']: q[k] for k in range(n_dof)}
        _, link_transforms, _ = compute_fk(
            links, joints, joint_angles=joint_angles, base_pos=base_pos, urdf_path=urdf_path
        )
        ee_pos = link_transforms[ee_link][:3, 3]
        ee_path.append(ee_pos.tolist())

    with open(scene_path, 'r') as f:
        scene_data = json.load(f)
    scene_data['ee_path'] = ee_path
    with open(scene_path, 'w') as f:
        json.dump(scene_data, f, indent=2)

    return True


def renumber_scenes(data_dir):
    """将剩余的场景文件按顺序重新编号"""
    files = sorted(glob.glob(os.path.join(data_dir, "env_*.json")))
    for new_id, old_path in enumerate(files):
        new_name = os.path.join(data_dir, f"env_{new_id:05d}.json")
        if old_path != new_name:
            # 更新文件内的 scene_id
            with open(old_path, 'r') as f:
                data = json.load(f)
            data['scene_id'] = new_id
            with open(old_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.rename(old_path, new_name)
    print(f"重新编号完成，共 {len(files)} 个场景")


def main():
    parser = argparse.ArgumentParser(description="验证场景可规划性，删除不可规划的场景")
    parser.add_argument('--start', type=int, default=None, help="起始场景 ID（含）")
    parser.add_argument('--end', type=int, default=None, help="结束场景 ID（不含）")
    parser.add_argument('--no-renumber', action='store_true', help="不重新编号")
    args = parser.parse_args()

    # 收集场景文件
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "env_*.json")))
    if not all_files:
        print("没有找到场景文件"); return

    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end or len(all_files)
        all_files = all_files[start:end]

    print(f"共 {len(all_files)} 个场景待验证")

    # 先做一次运动学后端检查，避免跑到后面才因为环境问题失败。
    # 解析 URDF（只需一次）
    links, joints = parse_urdf(URDF_PATH)
    ee_link = get_ee_link(joints)
    print(f"URDF 加载完成，ee_link={ee_link}")

    success_count = 0
    fail_count = 0
    deleted_files = []

    for i, scene_path in enumerate(all_files):
        scene_name = os.path.basename(scene_path)
        try:
            ok = validate_scene(links, joints, ee_link, scene_path, URDF_PATH)
        except Exception as e:
            print(f"  [{i+1}/{len(all_files)}] {scene_name}: 异常 - {e}")
            # 对环境/依赖问题直接终止，避免把大量场景误判为失败并删除。
            if "buildModelFromUrdf" in str(e) or "pinocchio" in str(e).lower():
                print("检测到 Pinocchio 环境异常，提前终止，未继续删除后续场景。")
                break
            ok = False

        if ok:
            success_count += 1
            print(f"  [{i+1}/{len(all_files)}] {scene_name}: 通过")
        else:
            fail_count += 1
            print(f"  [{i+1}/{len(all_files)}] {scene_name}: 失败，删除")
            deleted_files.append(scene_path)
            os.remove(scene_path)

    print(f"\n验证完成: 通过 {success_count}, 失败 {fail_count} (已删除)")

    # 重新编号
    if not args.no_renumber and fail_count > 0:
        renumber_scenes(DATA_DIR)


if __name__ == "__main__":
    main()
