# panda_realtime_sim.py
import pybullet as p
import pybullet_data
import numpy as np
import time, json, os

EE_LINK = 11
STATE_FILE = "control_state.json"
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1/240)
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

for j in range(p.getNumJoints(robot)):
    p.resetJointState(robot, j, 0.0)

velocity = np.zeros(3)

print("Simulation is running. Waiting for external updates...")

while True:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                ctrl = json.load(f)
            K, B = ctrl["K"], ctrl["B"]
            intent = ctrl["intent"]
        else:
            K, B, intent = 50, 30, "idle"

        state = p.getLinkState(robot, EE_LINK, computeLinkVelocity=1)
        pos = np.array(state[0])
        vel = np.array(state[6])
        acc = (-B * vel - K * pos) / 1.0
        velocity += acc * (1/240)
        target = pos + velocity * (1/240)
        angles = p.calculateInverseKinematics(robot, EE_LINK, target)
        for j in range(len(angles)):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=angles[j])
        p.stepSimulation()
        time.sleep(1/240)
    except Exception as e:
        print(e)
        time.sleep(0.1)

