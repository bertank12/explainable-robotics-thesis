import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
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

# Model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Decision plot for index 5
index = 5
pred_class = model.predict([X_test.iloc[index]])[0]

# SHAP value fix if extra dimension
shap_val = shap_values.values[index][pred_class]
if len(shap_val) == len(X.columns) + 1:
    shap_val = shap_val[:-1]  # remove base value if needed

shap.decision_plot(
    explainer.expected_value[pred_class],
    shap_val,
    X_test.iloc[index],
    feature_names=X.columns
)

plt.title("SHAP Decision Plot (Index 5)")
plt.tight_layout()
plt.savefig("shap_decision_plot.png")
plt.close()

print("Saved: shap_decision_plot.png")

