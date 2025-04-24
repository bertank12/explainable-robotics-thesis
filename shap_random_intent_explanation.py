import pandas as pd
import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

# Load patched dataset
files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

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

# Train model (CPU compatible)
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    tree_method="hist",  # safe fallback
    predictor="auto"
)
model.fit(X_train, y_train)

# Pick a random test sample
idx = random.randint(0, len(X_test) - 1)
sample = X_test.iloc[idx]
true_label = y_test[idx]
predicted_label = model.predict([sample])[0]

print(f"\nTrue class: {labels[true_label]}")
print(f"Predicted class: {labels[predicted_label]}", end=" ")
print("Correct" if true_label == predicted_label else "Incorrect")

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # list of arrays, one per class

shap_sample = shap_values[predicted_label][idx]  # (features,)
lines = [f"\nSHAP Explanation for predicted class '{labels[predicted_label]}':\n"]

for fname, fval, sval in zip(X.columns, sample, shap_sample):
    sign = "+" if sval >= 0 else "-"
    lines.append(f"  - Feature '{fname}' = {fval:.4f} → SHAP: {sign}{abs(sval):.4f}")

print("\n".join(lines))

