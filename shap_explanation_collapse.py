import pandas as pd
import shap
import xgboost as xgb
import numpy as np
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

# Select a sample of the "collapse" class
target_class = "collapse"
class_index = list(labels).index(target_class)
sample_index = np.where(y_test == class_index)[0][0]
sample_features = X_test.iloc[sample_index]
sample_shap = shap_values.values[sample_index, class_index, :]

# Sort SHAP values
sorted_indices = np.argsort(np.abs(sample_shap))[::-1]

# Explanation
lines = []
lines.append(f"SHAP Explanation for predicted intent: '{target_class}'\n")
lines.append("The model made this prediction mainly due to the following feature influences:\n")

for idx in sorted_indices[:3]:
    fname = X.columns[idx]
    fvalue = sample_features[fname]
    shapval = sample_shap[idx]
    direction = "increased" if shapval > 0 else "decreased"
    lines.append(f"- {fname} = {fvalue:.3f} → {direction} the confidence for '{target_class}' (SHAP: {shapval:.4f})")

# Save to file
with open("shap_explanation_collapse.txt", "w") as f:
    f.write("\n".join(lines))

print("Explanation saved to shap_explanation_collapse.txt")

