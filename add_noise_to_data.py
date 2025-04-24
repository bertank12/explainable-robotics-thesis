import pandas as pd
import numpy as np

# Noisy veri üretmek istediğimiz dosyalar
csv_files = [
    "panda_data_approach.csv", "panda_data_collapse.csv", "panda_data_grasp.csv",
    "panda_data_handshake.csv", "panda_data_idle.csv", "panda_data_pushaway.csv",
    "panda_data_retreat.csv", "panda_data_wave.csv"
]

# Noise seviyesi (standart sapma)
noise_std = 0.01

for file in csv_files:
    df = pd.read_csv(file)
    
    # Sadece pozisyon + hız için noise uygula
    for col in ["x", "y", "z", "vx", "vy", "vz"]:
        df[col] += np.random.normal(0, noise_std, size=len(df))
    
    # Yeni dosya ismi
    new_file = file.replace(".csv", "_noisy.csv")
    df.to_csv(new_file, index=False)
    print(f"Noisy dataset saved: {new_file}")

