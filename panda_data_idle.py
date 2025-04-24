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

# Link index of the end-effector (hand)
HAND_LINK_INDEX = 11
data = []

print("Collecting IDLE motion data... Robot remains stationary.")

for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240)

    # Get position and velocity of the end-effector
    state = p.getLinkState(robot_id, HAND_LINK_INDEX, computeLinkVelocity=1)
    if state:
        pos = state[0]
        vel = state[6]
    else:
        continue

    # Count contact points (if any)
    contact_force = len(p.getContactPoints(bodyA=robot_id))

    data.append({
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "vx": vel[0],
        "vy": vel[1],
        "vz": vel[2],
        "force": contact_force,
        "label": "idle"
    })

# Disconnect and save data
p.disconnect()
print("Data collection completed. Saving CSV...")

df = pd.DataFrame(data)
df.to_csv("panda_data_idle.csv", index=False)
print("File saved: panda_data_idle.csv")

