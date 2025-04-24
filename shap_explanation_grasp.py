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

X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Target class: grasp
target_class = "grasp"
class_index = list(labels).index(target_class)

# Find one sample of this class
sample_indices = np.where(y_test == class_index)[0]
sample_index = sample_indices[0]

# SHAP values and features
sample_features = X_test.iloc[sample_index]
shap_vals = shap_values.values[sample_index, class_index, :-1]

# Sort by importance
importance_order = np.argsort(np.abs(shap_vals))[::-1]

# Format explanation
lines = [f"SHAP Explanation for class '{target_class}':\n"]
for idx in importance_order:
    fname = X.columns[idx]
    fval = sample_features.iloc[idx]
    sval = shap_vals[idx]
    lines.append(f"  - Feature '{fname}' = {fval:.4f} → SHAP: {sval:+.4f}")

# Save to file
with open("shap_explanation_grasp.txt", "w") as f:
    f.write("\n".join(lines))

print("Explanation saved to shap_explanation_grasp.txt")

