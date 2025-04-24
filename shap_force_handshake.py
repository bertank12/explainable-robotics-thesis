import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Split
X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Filter only handshake class
class_name = "handshake"
class_index = np.where(labels == class_name)[0][0]
class_mask = (y_test == class_index)

# Extract relevant data
force_values = X_test[class_mask]["force"]
shap_force_values = shap_values[class_mask].values[:, class_index, X.columns.get_loc("force")]
color_feature = X_test[class_mask]["vx"]

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    force_values,
    shap_force_values,
    c=color_feature,
    cmap="coolwarm",
    edgecolor='k'
)
plt.colorbar(scatter, label="vx")
plt.xlabel("Force")
plt.ylabel("SHAP Value for Force")
plt.title("SHAP Force Impact for 'handshake'")
plt.grid(True)
plt.tight_layout()
plt.savefig("shap_force_handshake.png")
plt.close()

