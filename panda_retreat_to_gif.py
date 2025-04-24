import pybullet as p
import pybullet_data
import time
import imageio
import os

# GIF için frame'leri topla
frames = []
gif_filename = "retreat_simulation.gif"
os.makedirs("gif_frames", exist_ok=True)

# PyBullet GUI başlat
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

# Simülasyon boyunca hareket ve frame toplama
for i in range(100):
    # Basit bir hareket
    position = [0, 0, 0.1 + i * 0.001]
    p.resetBasePositionAndOrientation(robot_id, position, [0, 0, 0, 1])
    p.stepSimulation()

    # Kamera görüntüsü al
    width, height, rgb, *_ = p.getCameraImage(
        width=320, height=240,
        viewMatrix=p.computeViewMatrix(
            cameraEyePosition=[1, 1, 1],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        ),
        projectionMatrix=p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
        )
    )
    frames.append(rgb)

    time.sleep(1. / 60.)

p.disconnect()

# GIF olarak kaydet
imageio.mimsave(gif_filename, frames, duration=1 / 60.)
print(f"GIF saved as {gif_filename}")

