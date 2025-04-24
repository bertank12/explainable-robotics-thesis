import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Simulate realistic external data
n_samples = 1000
data = {
    "x": np.random.normal(loc=0.0, scale=0.5, size=n_samples),
    "y": np.random.normal(loc=0.0, scale=0.5, size=n_samples),
    "z": np.random.normal(loc=1.0, scale=0.5, size=n_samples),
    "vx": np.random.normal(loc=0.0, scale=0.1, size=n_samples),
    "vy": np.random.normal(loc=0.0, scale=0.1, size=n_samples),
    "vz": np.random.normal(loc=0.0, scale=0.1, size=n_samples),
    "force": np.random.normal(loc=10.0, scale=2.0, size=n_samples),
    "label": np.random.choice([
        "approach", "collapse", "grasp", "handshake",
        "idle", "pushaway", "retreat", "wave"
    ], size=n_samples)
}

external_df = pd.DataFrame(data)

# Save to CSV
file_path = "/home/ubuntu/Bertan_Kavak_Dissertation/external_data.csv"
external_df.to_csv(file_path, index=False)

file_path

