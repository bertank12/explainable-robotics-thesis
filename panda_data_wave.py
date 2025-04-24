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

# Load the environment and the robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

HAND_LINK_INDEX = 11
data = []
t = 0

print("Starting WAVE motion (hand raised with forward tilt)...")

for i in range(1000):
    t += 1./240
    p.stepSimulation()
    time.sleep(1./240)

    # Set initial wrist orientation (raised and rotated)
    p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=2.5)    # wrist roll
    p.setJointMotorControl2(robot_id, 6, p.POSITION_CONTROL, targetPosition=0.0)    # wrist pitch (initial)
    p.setJointMotorControl2(robot_id, 7, p.POSITION_CONTROL, targetPosition=-2.0)   # wrist yaw

    # Apply waving motion on wrist pitch
    wave_pos = 0.4 * math.sin(8 * t)
    p.setJointMotorControl2(robot_id, 6, p.POSITION_CONTROL, targetPosition=wave_pos)

    # Collect data
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
        "label": "wave"
    })

# Finalize and export
p.disconnect()
print("Data collected, saving to CSV...")

df = pd.DataFrame(data)
df.to_csv("panda_data_wave.csv", index=False)
print("Completed: panda_data_wave.csv")

