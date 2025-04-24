import pybullet as p
import pybullet_data
import os
import time
from PIL import Image
import numpy as np

# Frame'lerin kaydedileceği klasör
output_dir = "retreat_frames"
os.makedirs(output_dir, exist_ok=True)

# PyBullet başlat
physicsClient = p.connect(p.DIRECT)  # GUI yerine DIRECT
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")

# Robot yükle (örnek: kuka)
robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# Frame sayısı
num_frames = 100

# Hareket simülasyonu
for i in range(num_frames):
    # Sinüsle ileri-geri hareket
    target_pos = 0.5 * np.sin(i * 0.1)
    p.setJointMotorControl2(
        robotId,
        jointIndex=2,
        controlMode=p.POSITION_CONTROL,
        targetPosition=target_pos
    )

    p.stepSimulation()

    # Kamera görüntüsü
    width, height, rgbImg, depth, seg = p.getCameraImage(
        width=320,
        height=240,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # PNG olarak kaydet
    img = Image.fromarray(rgbImg)
    img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))

    time.sleep(1. / 60)

p.disconnect()
print(f"{num_frames} frame oluşturuldu: {output_dir}/")

