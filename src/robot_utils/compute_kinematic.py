import numpy as np
import pinocchio as pin
from robot_utils.urdf_to_geometry import parse_urdf
from robot_utils.rotation_matrix import make_transform


pinocchio_instances = {}  


def get_pinocchio_instance(urdf_path, package_dirs=None):
    """单例模式获取 PinocchioKinematics 实例"""
    if urdf_path not in pinocchio_instances:
        pinocchio_instances[urdf_path] = PinocchioKinematics(urdf_path, package_dirs)
    return pinocchio_instances[urdf_path]

class PinocchioKinematics:
    def __init__(self, urdf_path, package_dirs=None):

        # 构建模型
        self.model = pin.buildModel(urdf_path, package_dirs=package_dirs)
        self.data = self.model.createData()
        # 解析 URDF 几何信息
        self.links, self.joints = parse_urdf(urdf_path)


        # 筛选 1-DOF 关节 (revolute/continuous)
        self.joint_names = [] #name
        self.joint_q = [] #idx_q
        self.l_limits = [] #lower limits
        self.u_limits = [] #upper limits
        lower_limits = np.asarray(self.model.lowerPositionLimit, dtype=float)
        upper_limits = np.asarray(self.model.upperPositionLimit, dtype=float)


        for joint in self.model.joints[1:]:
            #from joint 1  , skip base link

            if joint.n_q != 1: 
                continue
            if joint.n_v == 0: 
                continue


            j_name = self.model.names[joint.id]
            q_idx= int(joint.idx_q)

            self.joint_names.append(j_name)
            self.joint_q.append(q_idx)

            low_limits = lower_limits[q_idx] if np.isfinite(lower_limits[q_idx]) else -np.pi
            high_limits = upper_limits[q_idx] if np.isfinite(upper_limits[q_idx]) else np.pi
            self.l_limits.append(low_limits)
            self.u_limits.append(high_limits)

        self.joint_q = np.asarray(self.joint_q, dtype=int)
        self.l_limits = np.asarray(self.l_limits, dtype=float)
        self.u_limits = np.asarray(self.u_limits, dtype=float)


    @staticmethod
    def se3_to_matrix(se3_obj) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = se3_obj.rotation
        T[:3, 3] = se3_obj.translation
        return T

    def compose_q(self, joint_angles=None, q=None):
        q_full = pin.neutral(self.model)

        if q is not None:

            q = np.asarray(q, dtype=float)
            if q.shape[0] == self.model.nq:
                q_full[:] = q
            elif q.shape[0] == len(self.q_indices):
                q_full[self.q_indices] = q
            else:
                raise ValueError("q 维度不匹配")
            return q_full

        if joint_angles is None: 
            return q_full
        for name, angle in joint_angles.items():
            if self.model.existJointName(name):
                jid = self.model.getJointId(name)
                joint = self.model.joints[jid]
                if joint.nq == 1:
                    q_full[int(joint.idx_q)] = float(angle)
        return q_full




    def forward_kinematics(self, joint_angles=None, q=None, base_pos=(0.0, 0.0, 0.0)):
        q_full = self.compose_q(joint_angles=joint_angles, q=q)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        T_base = np.eye(4)
        T_base[:3, 3] = np.asarray(base_pos, dtype=float)

        link_transforms = {"base": T_base.copy()}
        for link_name in self.links.keys():
            if self.model.existFrame(link_name):
                frame_id = self.model.getFrameId(link_name)
                T_link = self.se3_to_matrix(self.data.oMf[frame_id])
                link_transforms[link_name] = T_base @ T_link

        # 构造视觉和碰撞几何体（保持与旧代码兼容）
        world_visuals = []
        world_collisions = []
        # 这里简化处理，实际如需详细几何体需遍历 self.links
        
        return world_visuals, link_transforms, world_collisions



    def inverse_kinematics(self, target_pos, ee_link, base_pos=(0.0, 0.0, 0.0), 
                           q_init=None, max_iter=200, tol=1e-3, damping=1e-2, 
                           step_scale=1.0, joint_limits=None):
        if not self.model.existFrame(ee_link):
            raise ValueError(f"末端 link 不存在: {ee_link}")
        ee_fid = self.model.getFrameId(ee_link)

        target_pos = np.asarray(target_pos, dtype=float)
        base_pos = np.asarray(base_pos, dtype=float)
        local_target = target_pos - base_pos

        n_dof = len(self.q_indices)
        if q_init is None:
            q_dof = 0.5 * (self.lower_limits + self.upper_limits)
        else:
            q_init = np.asarray(q_init, dtype=float)
            if q_init.shape[0] == self.model.nq:
                q_dof = q_init[self.q_indices].copy()
            elif q_init.shape[0] == n_dof:
                q_dof = q_init.copy()
            else:
                raise ValueError("q_init 维度不匹配")

        if joint_limits is not None:
            jl = np.asarray(joint_limits, dtype=float)
            lo, hi = jl[:, 0], jl[:, 1]
        else:
            lo, hi = self.lower_limits, self.upper_limits

        eye3 = np.eye(3)
        for _ in range(max_iter):
            q_full = pin.neutral(self.model)
            q_full[self.q_indices] = q_dof

            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)

            ee_pos = self.data.oMf[ee_fid].translation
            err = local_target - ee_pos

            if np.linalg.norm(err) < tol:
                return q_dof, True

            J6 = pin.computeFrameJacobian(self.model, self.data, q_full, ee_fid, 
                                          pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J = J6[:3, self.q_indices]
            JJT = J @ J.T + damping * eye3
            dq = J.T @ np.linalg.solve(JJT, err)

            q_dof = q_dof + step_scale * dq
            q_dof = np.clip(q_dof, lo, hi)

        return q_dof, False

    def inverse_kinematics_multi_start(self, target_pos, ee_link, base_pos=(0.0, 0.0, 0.0),
                                       n_starts=20, max_iter=200, tol=1e-3, damping=1e-2,
                                       step_scale=1.0, joint_limits=None, collision_checker=None):
        n_dof = len(self.q_indices)
        if joint_limits is not None:
            jl = np.asarray(joint_limits, dtype=float)
            lo, hi = jl[:, 0], jl[:, 1]
        else:
            lo, hi = self.lower_limits, self.upper_limits

        best_q = None
        best_err = np.inf

        for i in range(n_starts):
            q0 = 0.5 * (lo + hi) if i == 0 else lo + np.random.rand(n_dof) * (hi - lo)
            
            q_sol, success = self.inverse_kinematics(
                target_pos, ee_link, base_pos, q0, max_iter, tol, damping, step_scale,
                np.column_stack([lo, hi])
            )
            if q_sol is None: continue

            _, link_tf, _ = self.forward_kinematics(q=q_sol, base_pos=base_pos)
            if ee_link not in link_tf: continue
            err = np.linalg.norm(np.asarray(target_pos) - link_tf[ee_link][:3, 3])

            if err < best_err:
                best_err = err
                best_q = q_sol.copy()

            if success:
                if collision_checker is None: return q_sol, True
                if collision_checker(q_sol): return q_sol, True

        return best_q, (best_err < tol)

# ───────── 兼容层：供外部调用的函数 ─────────

def compute_fk(links, joints, joint_angles=None, q=None, base_pos=(0.0, 0.0, 0.0), urdf_path=None):
    """
    兼容接口：计算正运动学
    如果传入了 urdf_path，则使用 Pinocchio 计算；否则尝试使用旧逻辑（此处简化为必须使用 Pinocchio）
    """
    # 这里假设你总是传入 urdf_path 或者在外部已经配置好
    # 为了兼容 generate_env_raw.py 的调用方式，我们需要动态获取实例
    # 注意：generate_env_raw.py 中已经解析了 links/joints，但为了计算 FK 我们需要 Pinocchio 模型
    # 这里做一个假设：URDF_PATH 在外部是已知的，或者我们需要修改调用方式。
    # 但为了最小改动，我们假设这里能获取到 URDF 路径。
    # *修正*：generate_env_raw.py 并没有传 urdf_path。
    # 因此，我们需要修改 generate_env_raw.py 或者在这里做一个全局的 URDF 路径假设。
    # 最稳妥的方式：修改 generate_env_raw.py 传入 urdf_path，或者在这里硬编码路径（不推荐）。
    
    # 鉴于 generate_env_raw.py 的调用签名是 compute_fk(links, joints, joint_angles=ja, base_pos=base_pos)
    # 我们无法在这里获取 URDF 路径。
    # 解决方案：我们需要修改 generate_env_raw.py，让它把 Pinocchio 实例或者 URDF 路径传进来。
    # 但为了让你现在的代码跑通，我将在 generate_env_raw.py 的修改版中解决这个问题。
    
    # 临时方案：这里抛出一个异常提示需要 URDF 路径，或者我们假设一个全局变量
    # 实际上，最好的办法是让 generate_env_raw.py 使用 PinocchioKinematics 类直接调用。
    # 但为了维持函数式调用，我们假设用户会传入 urdf_path 关键字参数，或者我们在这里报错。
    
    # 让我们采用一种更灵活的方式：如果没传 urdf_path，尝试从 joints 推断（很难），
    # 或者我们直接修改 generate_env_raw.py 的导入和使用方式。
    
    # 鉴于上下文，我决定在 generate_env_raw.py 中直接实例化 PinocchioKinematics。
    # 这里保留函数定义以兼容旧代码，但逻辑上依赖 urdf_path。
    if urdf_path is None:
        # 尝试从环境变量或默认路径获取，或者报错
        # 这里为了演示，假设有一个全局变量或者你需要修改调用方
        raise ValueError("compute_fk 需要 urdf_path 参数来初始化 Pinocchio 模型")
        
    kin = get_pinocchio_instance(urdf_path)
    return kin.forward_kinematics(joint_angles=joint_angles, q=q, base_pos=base_pos)

def compute_ik(links, joints, target_pos, base_pos=(0.0, 0.0, 0.0), 
               joint_limits=None, n_starts=50, tol=0.01, ee_link=None, 
               collision_checker=None, urdf_path=None):
    """
    兼容接口：计算逆运动学
    """
    if urdf_path is None:
         raise ValueError("compute_ik 需要 urdf_path 参数")
         
    kin = get_pinocchio_instance(urdf_path)
    return kin.inverse_kinematics_multi_start(
        target_pos=target_pos,
        ee_link=ee_link,
        base_pos=base_pos,
        n_starts=n_starts,
        tol=tol,
        joint_limits=joint_limits,
        collision_checker=collision_checker
    )