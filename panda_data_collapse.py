import pybullet as p
import pybullet_data
import time
import pandas as pd

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./240)

# Load environment and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Index of the end-effector (hand)
HAND_LINK_INDEX = 11
data = []

print("COLLAPSE simulation started: robot arm collapses in place...")

# Release all joints by disabling motor forces
for joint in range(p.getNumJoints(robot_id)):
    p.setJointMotorControl2(robot_id, joint, controlMode=p.VELOCITY_CONTROL, force=0)

# Run simulation
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240)

    state = p.getLinkState(robot_id, HAND_LINK_INDEX, computeLinkVelocity=1)
    if state:
        pos = state[0]
        vel = state[6]
    else:
        continue

    contact_force = len(p.getContactPoints(bodyA=robot_id))

    data.append({
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "vx": vel[0],
        "vy": vel[1],
        "vz": vel[2],
        "force": contact_force,
        "label": "collapse"
    })

# Disconnect and save
p.disconnect()
print("Data collected, saving to CSV...")

df = pd.DataFrame(data)
df.to_csv("panda_data_collapse.csv", index=False)
print("Completed: panda_data_collapse.csv")

