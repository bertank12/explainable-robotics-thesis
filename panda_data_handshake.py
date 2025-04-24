import pybullet as p
import pybullet_data
import time
import pandas as pd
import math

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./240)

# Load environment and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

HAND_LINK_INDEX = 11
data = []
t = 0

print("Starting HANDSHAKE: arm is fixed, wrist performs oscillatory motion")

for i in range(1000):
    t += 1./240
    p.stepSimulation()
    time.sleep(1./240)

    # Set arm to handshake pose (fixed)
    p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=0.8)
    p.setJointMotorControl2(robot_id, 2, p.POSITION_CONTROL, targetPosition=-0.4)
    p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=-1.5)
    p.setJointMotorControl2(robot_id, 4, p.POSITION_CONTROL, targetPosition=1.0)
    p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=2.0)
    p.setJointMotorControl2(robot_id, 7, p.POSITION_CONTROL, targetPosition=-8.0)

    # Oscillatory movement in the wrist (joint 6)
    shake = 0.3 * math.sin(10 * t)
    p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=shake)

    # Record data
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
        "label": "handshake"
    })

# Disconnect and export data
p.disconnect()
print("Data collected. Saving to CSV...")

df = pd.DataFrame(data)
df.to_csv("panda_data_handshake.csv", index=False)
print("Completed: panda_data_handshake.csv")

