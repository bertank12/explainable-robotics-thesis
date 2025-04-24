import pybullet as p
import pybullet_data
import time
import pandas as pd

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1/240)

# Load robot and environment
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Prevent robot from rotating
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

print("STARTING CONTROLLED APPROACH MOVEMENT...")

# Target joint positions
target_joint_1 = 1.0
target_joint_3 = 1.0
target_joint_5 = -2.0

for i in range(1000):
    p.stepSimulation()
    time.sleep(1/240)

    # Read current joint positions
    current_1 = p.getJointState(robot_id, 1)[0]
    current_3 = p.getJointState(robot_id, 3)[0]
    current_5 = p.getJointState(robot_id, 5)[0]

    # Gradually move joints toward target
    speed = 0.02
    new_1 = current_1 + min(speed, target_joint_1 - current_1)
    new_3 = current_3 + min(speed, target_joint_3 - current_3)
    new_5 = current_5 + min(speed, target_joint_5 - current_5)

    # Apply updated joint positions
    p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=new_1)
    p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=new_3)
    p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=new_5)

    # Fix all other joints
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
            "label": "approach"
        })

while True:
    for i in range(240): 
        p.stepSimulation()
        time.sleep(1./240.)


print("Movement complete. Press Enter to close.")
input()

# Disconnect and save data
p.disconnect()
print("Data collection complete, saving to CSV...")

df = pd.DataFrame(data)
df.to_csv("panda_data_approach.csv", index=False)
print("Completed: panda_data_approach.csv")


