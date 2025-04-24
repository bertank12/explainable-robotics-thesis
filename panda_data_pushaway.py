import pybullet as p
import pybullet_data
import time
import pandas as pd

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./240)

# Load the environment and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Fix the robot base to prevent rotation
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

print("Starting PUSH-AWAY movement using joints 1, 3, and 5 (negative direction)...")

for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240)

    if i < 300:
        t = i / 300
        p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=1.2 * t)
        p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=2.8 * t, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=-2.5 * t, maxVelocity=1.5)
    else:
        p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=1.2)
        p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=-0.8, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=2.5, maxVelocity=1.5)

    # Keep all other joints fixed
    for joint in range(p.getNumJoints(robot_id)):
        if joint not in [1, 3, 5]:
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
            "label": "pushaway"
        })

# Finalize
p.disconnect()
print("Data collected, saving to CSV...")

df = pd.DataFrame(data)
df.to_csv("panda_data_pushaway.csv", index=False)
print("Completed: panda_data_pushaway.csv")

