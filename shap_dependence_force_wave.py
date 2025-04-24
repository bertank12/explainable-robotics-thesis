import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Patched CSV dosyaları
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
labels = le.classes_.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

print("SHAP shape:", shap_values.values.shape)
print("Labels:", labels)

class_index = list(labels).index("wave")
wave_indices = np.where(y_test == class_index)[0]
shap_vals_wave = shap_values.values[wave_indices, :, class_index]  # DÜZELTİLDİ

X_wave = X_test.iloc[wave_indices]

# Doğruluk kontrolü
assert len(X_wave) == len(shap_vals_wave), "Veri uzunlukları uyuşmuyor."

# Grafik çizimi
plt.figure()
plt.scatter(X_wave["force"], shap_vals_wave[:, X.columns.get_loc("force")], c=X_wave["vx"], cmap="coolwarm")
plt.xlabel("Force")
plt.ylabel("SHAP (force)")
plt.title("SHAP Dependence Plot – Force vs. vx (wave)")
plt.colorbar(label="vx")
plt.tight_layout()
plt.savefig("shap_dependence_force_wave.png")
plt.close()

