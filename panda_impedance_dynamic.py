import pybullet as p
import pybullet_data
import time
import numpy as np

# Constants
MASS = 1.0            # Virtual mass
DT = 1.0 / 240.0      # Simulation time step
INITIAL_POS = [0, 0, 0.5]

# Mapping: Intent → (Stiffness K, Damping B)
IMPEDANCE_MAP = {
    "approach": (20, 10),
    "retreat": (100, 40),
    "idle": (50, 50),
}

# Select the current intent (can be changed)
intent = "retreat"  # Options: "approach", "retreat", "idle"
K, B = IMPEDANCE_MAP[intent]

# Start PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")
sphere = p.loadURDF("sphere2.urdf", INITIAL_POS, useFixedBase=False)

# Initial state
position = np.array(INITIAL_POS)
velocity = np.array([0.0, 0.0, 0.0])

# Simulation loop
for i in range(1000):
    p.stepSimulation()
    time.sleep(DT)

    # Current state
    pos_now = np.array(p.getBasePositionAndOrientation(sphere)[0])
    vel_now = np.array(p.getBaseVelocity(sphere)[0])

    # External force (applied manually during certain time steps)
    if 100 < i < 300:
        force = np.array([100.0, 0.0, 0.0])  # Push from the side
    else:
        force = np.array([0.0, 0.0, 0.0])

    # Impedance control equation: a = (F - Bv - Kx) / M
    acceleration = (force - B * vel_now - K * (pos_now - INITIAL_POS)) / MASS

    # Update velocity and position
    velocity += acceleration * DT
    position += velocity * DT

    # Apply updated state
    p.resetBasePositionAndOrientation(sphere, position.tolist(), [0, 0, 0, 1])
    p.resetBaseVelocity(sphere, velocity.tolist(), [0, 0, 0])

p.disconnect()

