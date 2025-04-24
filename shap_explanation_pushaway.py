import pandas as pd
import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load patched dataset
files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Prepare features and labels
X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    tree_method="hist",        # CPU tabanlı
    predictor="auto"
)

model.fit(X_train, y_train)

# Explain with SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Select a sample for class 'pushaway'
class_name = "pushaway"
class_index = list(labels).index(class_name)
sample_index = np.where(y_test == class_index)[0][0]
sample_features = X_test.iloc[sample_index]
sample_shap = shap_values.values[sample_index, class_index]

# Format explanation
lines = [f"SHAP Explanation for class '{class_name}':\n"]
for fname, val, sval in zip(X.columns, sample_features, sample_shap):
    sign = "+" if sval >= 0 else "-"
    lines.append(f"  - Feature '{fname}' = {val:.4f} → SHAP: {sign}{abs(sval):.4f}")

# Save to file
with open("shap_explanation_pushaway.txt", "w") as f:
    f.write("\n".join(lines))

