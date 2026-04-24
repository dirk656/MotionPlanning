import pybullet as p
import pybullet_data
import time
import numpy as np

# ==========================================
# 1. 初始化环境
# ==========================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 加载地面
planeId = p.loadURDF("plane.urdf")

# 加载机械臂
robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
robot_base_center = np.array([0.0, 0.0, 0.0])
sphere_radius = 0.5

# ==========================================
# 2. 创建动态障碍物
# ==========================================
# 我们创建一个红色的盒子作为障碍物
box_size = 0.2  # 盒子边长 20cm
# 视觉属性：红色 (RGBA), 透明度 1
visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[box_size/2]*3, rgbaColor=[1, 0, 0, 1])
# 碰撞属性：用于物理碰撞检测
collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_size/2]*3)

# 加载障碍物 (初始位置在半径 0.5m 球内)
obstacle_id = p.createMultiBody(
    baseMass=0,  # 质量为 0 表示它是静态的（或者你可以设为 1 让它受重力掉落）
    baseCollisionShapeIndex=collision_shape_id,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=[0.3, 0, 0.3]
)

print(f"障碍物 ID: {obstacle_id}")

# 获取 Franka 7 轴手臂关节信息
num_joints = p.getNumJoints(robotId)
joint_indices = list(range(7)) 

# ==========================================
# 3. 仿真主循环
# ==========================================
try:
    t = 0  # 时间计数器
    while True:
        t += 0.02  # 时间步进
        
        # --- A. 更新障碍物位置 (动态逻辑) ---
        # 让障碍物在 X 轴方向做正弦往复运动
        # 范围：0.3 到 0.9，频率：1Hz
        new_x = 0.6 + 0.3 * np.sin(t)
        # 保持 Y 和 Z 不变
        candidate_pos = np.array([new_x, 0.0, 0.5])

        # 将障碍物中心位置限制在以机械臂底座为球心、半径 0.5m 的球内。
        offset = candidate_pos - robot_base_center
        distance = np.linalg.norm(offset)
        if distance > sphere_radius:
            candidate_pos = robot_base_center + (offset / distance) * sphere_radius

        new_pos = candidate_pos.tolist()
        
        # 重置障碍物位置
        p.resetBasePositionAndOrientation(obstacle_id, new_pos, [0, 0, 0, 1])

        # --- B. 机械臂控制逻辑 ---
        # 这里演示位置控制，你可以换成 IK 控制
        target_positions = [0, -0.5, 0, -1.0, 0, 1.5, 0] 
        
        p.setJointMotorControlArray(
            robotId, 
            joint_indices, 
            p.POSITION_CONTROL, 
            targetPositions=target_positions,
            positionGains=[0.1] * 7,
            velocityGains=[1.0] * 7
        )

        # --- C. 物理步进 ---
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    pass

p.disconnect()
