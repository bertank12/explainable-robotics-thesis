import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Modeli yeniden eğit (veya daha önce eğitilmiş modeli yükle)
files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df_train = pd.concat([pd.read_csv(f) for f in files])
X_train = df_train[["x", "y", "z", "vx", "vy", "vz", "force"]]
y_train = df_train["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", tree_method="hist")
model.fit(X_train, y_encoded)

# --- External Data Test ---
external_df = pd.read_csv("external_data.csv")
X_ext = external_df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y_ext = le.transform(external_df["label"])  # LabelEncoder uyumlu olmalı

y_pred = model.predict(X_ext)

acc = accuracy_score(y_ext, y_pred)
print(f"Accuracy on external data: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_ext, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_ext, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix on External Data\nAccuracy: {acc:.4f}")
plt.tight_layout()
plt.show()

