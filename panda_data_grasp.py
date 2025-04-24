import pybullet as p
import pybullet_data
import time
import pandas as pd

# Setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1/240)

# Load robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Fix base
p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)

HAND_LINK_INDEX = 11
data = []

print("Starting GRASP: move to approach position, then close gripper")

# Phase 1: move to approach pose (first 300 steps)
for i in range(1000):
    p.stepSimulation()
    time.sleep(1/240)

    if i < 300:
        # Move arm to approach position
        t = i / 300
        p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=1.3 * t)
        p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=1.0 * t)
        p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=1.5 * t)

        # Gripper open during movement
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04)

    else:
        # Maintain approach pose
        p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=1.3)
        p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=1.0)
        p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=1.5)

        # Gripper gradually closes
        grasp_amount = max(0.0, 0.04 - 0.0002 * (i - 300))
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=grasp_amount)
        p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=grasp_amount)

    # Other joints fixed
    for joint in range(p.getNumJoints(robot_id)):
        if joint not in [1, 3, 5, 9, 10]:
            p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=0.0)

    # Record data
    state = p.getLinkState(robot_id, HAND_LINK_INDEX, computeLinkVelocity=1)
    if state:
        pos = state[0]
        vel = state[6]
        force = len(p.getContactPoints(bodyA=robot_id))
        data.append({
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "vx": vel[0],
            "vy": vel[1],
            "vz": vel[2],
            "force": force,
            "label": "grasp"
        })

# Disconnect and save
p.disconnect()
print("Saving to CSV...")

df = pd.DataFrame(data)
df.to_csv("panda_data_grasp.csv", index=False)
print("Completed: panda_data_grasp.csv")

