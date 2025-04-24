import pandas as pd
import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load all patched CSV files
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Select sample for class
class_name = "approach"
class_index = list(labels).index(class_name)
sample_indices = np.where(y_test == class_index)[0]
sample_index = sample_indices[0]
sample_features = X_test.iloc[sample_index]
shap_vals = shap_values.values[sample_index, class_index, :]

# Most influential features
sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
top_features = [(X.columns[i], sample_features[i], shap_vals[i]) for i in sorted_idx[:3]]

# Always include vx, vy, vz
velocity_features = ["vx", "vy", "vz"]
velocity_details = [(v, sample_features[v], shap_vals[X.columns.get_loc(v)]) for v in velocity_features]

# Write explanation
lines = []
lines.append(f"SHAP-Based Explanation for Human-Robot Intent: '{class_name}'\n")
lines.append("The most influential features on this prediction were:\n")

for fname, val, shapval in top_features:
    direction = "increased" if shapval > 0 else "decreased"
    lines.append(f"- '{fname}' = {val:.3f} → {direction} confidence in predicting '{class_name}'")

lines.append("\nAdditional motion dynamics:\n")
for fname, val, shapval in velocity_details:
    impact = "high" if abs(shapval) > 0.1 else "low"
    lines.append(f"- '{fname}' = {val:.3f} → {impact} impact (SHAP = {shapval:.3f})")

explanation = "\n".join(lines)

# Save to file
with open("shap_explanation_approach_detailed.txt", "w") as f:
    f.write(explanation)

print("Detailed SHAP explanation saved as 'shap_explanation_approach_detailed.txt'")

