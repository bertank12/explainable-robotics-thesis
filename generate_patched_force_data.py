import pandas as pd
import numpy as np
import os

# Kuvvet aralıkları: sınıfa göre (mean, std)
force_config = {
    "approach": (8.0, 2.0),
    "collapse": (12.0, 3.0),
    "grasp": (10.0, 2.0),
    "handshake": (9.0, 2.0),
    "idle": (4.0, 1.0),
    "pushaway": (11.0, 2.5),
    "retreat": (13.0, 3.0),
    "wave": (7.0, 1.5),
}

# Düzeltilecek dosyalar
files_to_patch = [
    "panda_data_approach.csv", "panda_data_approach_noisy.csv",
    "panda_data_grasp.csv", "panda_data_grasp_noisy.csv",
    "panda_data_handshake.csv", "panda_data_handshake_noisy.csv",
    "panda_data_idle.csv", "panda_data_idle_noisy.csv",
    "panda_data_pushaway.csv", "panda_data_pushaway_noisy.csv",
    "panda_data_retreat.csv", "panda_data_retreat_noisy.csv",
    "panda_data_wave.csv", "panda_data_wave_noisy.csv"
]

for file in files_to_patch:
    try:
        df = pd.read_csv(file)
        if "force" in df.columns and (df["force"] == 0.0).all():
            class_label = df["label"].iloc[0]
            mean, std = force_config.get(class_label, (10.0, 2.0))
            new_force = np.clip(np.random.normal(loc=mean, scale=std, size=len(df)), 0.1, None)
            df["force"] = new_force
            output_file = file.replace(".csv", "_patched.csv")
            df.to_csv(output_file, index=False)
            print(f"Patched and saved: {output_file}")
        else:
            print(f"Skipped (already contains non-zero force): {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

