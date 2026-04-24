import pybullet as p
import pybullet_data

import time 
import numpy as np

from pybullet.env.tools import generate_pos

#init 
client_id = p.connect(p.GUI)
if client_id < 0:
    client_id = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
num_joints = p.getNumJoints(robotId)
joints_indices = list(range(num_joints))
end_effector = 6


#main 

start_pos , goal_pos = generate_pos(1.0)
while p.isConnected(client_id):
    j_qs = p.calculateInverseKinematics(robotId, end_effector, start_pos.tolist())
    p.setJointMotorControlArray(robotId, joints_indices, p.POSITION_CONTROL, targetPositions=j_qs)
    for _ in range(1000):
        if not p.isConnected(client_id):
            break
        p.stepSimulation()
        time.sleep(1./240.)



if p.isConnected(client_id):
    p.disconnect(client_id)


