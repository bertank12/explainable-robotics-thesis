import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load patched datasets
files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Split features and target
X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Select index
index = 5
pred_class = model.predict([X_test.iloc[index]])[0]

# Generate interactive force plot
shap.initjs()
force_plot = shap.plots.force(
    explainer.expected_value[pred_class],
    shap_values.values[index][pred_class][:-1],  # Son sütunu çıkar!
    features=X_test.iloc[index],
    feature_names=X.columns
)

# Save HTML
with open("shap_force_plot.html", "w") as f:
    f.write(shap.getjs())
    f.write(force_plot.html())

