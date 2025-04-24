import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load patched CSVs
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# SHAP explain
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Select sample for 'handshake'
class_index = list(labels).index("handshake")
handshake_indices = np.where(y_test == class_index)[0]
if len(handshake_indices) == 0:
    raise ValueError("No samples found for 'handshake'")
i = handshake_indices[0]

sample_features = X_test.iloc[i]
sample_shap = shap_values.values[i, :, class_index]

# Format explanation
lines = [f"SHAP Explanation for class 'handshake':\n"]
for fname, val, shap_val in sorted(zip(X.columns, sample_features, sample_shap), key=lambda x: -abs(x[2])):
    lines.append(f"  - Feature '{fname}' = {val:.4f} → SHAP: {shap_val:+.4f}")

# Save to file
with open("shap_explanation_handshake.txt", "w") as f:
    f.write("\n".join(lines))

print("Saved: shap_explanation_handshake.txt")

