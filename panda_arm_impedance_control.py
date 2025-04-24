import pybullet as p
import pybullet_data
import numpy as np
import time

# Constants
MASS = 1.0  # virtual mass of the end-effector
DT = 1.0 / 240.0
EE_LINK_INDEX = 11  # end-effector link

# Intent-based impedance parameters
IMPEDANCE_PARAMS = {
    "approach": (20, 10),
    "idle": (60, 50),     # slightly stiffer
    "retreat": (150, 80)  # much stiffer
}


# Choose intent
intent = "retreat"  # options: "approach", "retreat", "idle"
K, B = IMPEDANCE_PARAMS[intent]

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Reset joint positions
for j in range(p.getNumJoints(robot)):
    p.resetJointState(robot, j, targetValue=0.0)

# Get initial end-effector position
init_state = p.getLinkState(robot, EE_LINK_INDEX)
init_pos = np.array(init_state[0])
velocity = np.zeros(3)

for i in range(1000):
    p.stepSimulation()
    time.sleep(DT)

    # Get current position & velocity
    state = p.getLinkState(robot, EE_LINK_INDEX, computeLinkVelocity=1)
    pos = np.array(state[0])
    vel = np.array(state[6])

    # External force
    if 100 < i < 300:
        F_ext = np.array([15.0, 0.0, 0.0])  # push in x direction
    else:
        F_ext = np.array([0.0, 0.0, 0.0])

    # Impedance equation
    acc = (F_ext - B * vel - K * (pos - init_pos)) / MASS
    velocity += acc * DT
    delta_pos = velocity * DT

    # Apply inverse kinematics to move end-effector
    target_pos = pos + delta_pos
    joint_angles = p.calculateInverseKinematics(robot, EE_LINK_INDEX, target_pos)

    # Apply joint angles
    for j in range(len(joint_angles)):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=joint_angles[j])

p.disconnect()

