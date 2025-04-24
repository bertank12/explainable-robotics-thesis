import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load patched data
files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Features and label
X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Retreat only
retreat_index = list(labels).index("retreat")
retreat_mask = y_test == retreat_index

shap_force_vals = shap_values[retreat_mask].values[:, retreat_index, X.columns.get_loc("force")]
force_vals = X_test[retreat_mask]["force"]
vx_vals = X_test[retreat_mask]["vx"]

plt.figure(figsize=(8, 6))
sc = plt.scatter(force_vals, shap_force_vals, c=vx_vals, cmap="viridis", edgecolor="k")
plt.colorbar(sc, label="vx")
plt.xlabel("Force")
plt.ylabel("SHAP Value (force)")
plt.title("SHAP Force vs. vx for Retreat Class")
plt.tight_layout()
plt.savefig("shap_dependence_retreat_vx.png")
plt.close()

