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

# Seçilen sınıf (örnek: "retreat")
class_index = list(labels).index("retreat")

# Dependence plot için sadece seçilen sınıfın shap değerleri
shap_vals_class = shap_values.values[:, class_index, :-1]

# Plot: force vs vx
shap.dependence_plot(
    "force",
    shap_vals_class,
    X_test,
    interaction_index="vx",
    show=False
)

plt.title("SHAP Dependence Plot (Force vs. vx) – Class: retreat")
plt.tight_layout()
plt.savefig("shap_dependence_force_fixed.png")
plt.close()

