import pandas as pd
import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data_files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)

# Features and labels
X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Select a sample with label "idle"
class_name = "idle"
class_index = list(labels).index(class_name)
sample_indices = np.where(y_test == class_index)[0]
sample_index = sample_indices[0]

sample_features = X_test.iloc[sample_index]
sample_shap = shap_values.values[sample_index, class_index, :]

# Format explanation
lines = [f"SHAP Explanation for class '{class_name}':", ""]
for i, (fname, val, shap_val) in enumerate(zip(X.columns, sample_features, sample_shap)):
    sign = "+" if shap_val >= 0 else "-"
    lines.append(f"  - Feature '{fname}' = {val:.4f} → SHAP: {sign}{abs(shap_val):.4f}")

# Save
with open("shap_explanation_idle.txt", "w") as f:
    f.write("\n".join(lines))

print("SHAP explanation for 'idle' saved to shap_explanation_idle.txt")

