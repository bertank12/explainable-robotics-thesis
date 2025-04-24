import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
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

# Generate SHAP dependence plot for each class
for class_index, class_name in enumerate(labels):
    print(f"Generating SHAP dependence plot for class: {class_name}")
    shap_vals_class = shap_values.values[:, class_index, :-1]  # drop offset column if exists

    plt.figure()
    shap.dependence_plot(
        "force",
        shap_vals_class,
        X_test,
        interaction_index="vx",
        show=False
    )
    plt.title(f"SHAP Dependence Plot (Force vs. vx) – Class: {class_name}")
    plt.tight_layout()
    plt.savefig(f"shap_dependence_force_{class_name}.png")
    plt.close()

