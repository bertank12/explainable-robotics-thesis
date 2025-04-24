import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
files = [
    "panda_data_approach.csv", "panda_data_collapse.csv", "panda_data_grasp.csv",
    "panda_data_handshake.csv", "panda_data_idle.csv", "panda_data_pushaway.csv",
    "panda_data_retreat.csv", "panda_data_wave.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Inject JS
shap.initjs()

# Select instance
index = 5
pred_class = model.predict([X_test.iloc[index]])[0]

# Fix: drop last element if SHAP adds base_value into SHAP array
shap_vals = shap_values.values[index][pred_class][:-1]  # <== the fix!

# Generate force plot
force_plot = shap.force_plot(
    explainer.expected_value[pred_class],
    shap_vals,
    X_test.iloc[index],
    feature_names=X.columns,
    matplotlib=False
)

# Save as full HTML with JS support
with open("shap_force_plot_full.html", "w") as f:
    f.write(shap.getjs())         # embed JS library
    f.write(force_plot.html())    # embed plot

print("Force plot saved: shap_force_plot_full.html")

