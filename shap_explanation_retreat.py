import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Patched datasetler
data_files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]

# Veri setlerini birleştir
df = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)

X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reset index — bu adım hatayı engeller
X_test = X_test.reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)

# Model eğit
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# SHAP hesapla
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# "retreat" sınıfı için örnek bir açıklama oluştur
class_name = "retreat"
class_index = list(labels).index(class_name)

# İlk "retreat" örneğini al
sample_index = np.where(y_test == class_index)[0][0]
sample_features = X_test.loc[sample_index]
sample_shap_values = shap_values.values[sample_index, class_index, :]

# Açıklamayı yazıya dök
contributions = list(zip(X.columns, sample_features, sample_shap_values))
contributions_sorted = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)

lines = []
lines.append(f"SHAP explanation for predicted class: {class_name}\n")
for name, value, shap_val in contributions_sorted:
    sign = "↑" if shap_val > 0 else "↓"
    lines.append(f"- {name}: {value:.3f} → {sign} contribution: {shap_val:.4f}")

# Kaydet
with open("shap_retreat_explanation.txt", "w") as f:
    f.write("\n".join(lines))

print("Explanation saved to shap_retreat_explanation.txt")

