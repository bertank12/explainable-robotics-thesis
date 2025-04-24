import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

# Patched CSV'leri oku
data_files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]

df_list = [pd.read_csv(file) for file in data_files]
data = pd.concat(df_list, ignore_index=True)

# Özellikleri ve etiketleri ayır
X = data[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = data["label"]

labels = sorted(y.unique().tolist())
label_map = {label: idx for idx, label in enumerate(labels)}
y_encoded = y.map(label_map)

# Eğitim / test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# XGBoost eğit
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Değerlendirme
y_pred = model.predict(X_test)
print("Model accuracy:", np.mean(y_pred == y_test))
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# SHAP analiz
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

features_np = X_test.to_numpy()
feature_names = X_test.columns.tolist()

# Bar grafiği için ortalama SHAP
mean_abs_shap = np.abs(shap_values.values[:, :, :-1]).mean(axis=(0, 1))
shap_order = np.argsort(mean_abs_shap)[::-1]

plt.figure()
plt.barh(
    [feature_names[i] for i in shap_order],
    mean_abs_shap[shap_order]
)
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.savefig("shap_feature_importance_bar.png")
plt.close()

# SHAP summary plot (örnek: approach sınıfı)
class_index = label_map["approach"]
shap_vals_class = shap_values.values[:, class_index, :-1]

shap.summary_plot(
    shap_vals_class,
    features=features_np,
    feature_names=feature_names,
    show=False
)
plt.title(f"SHAP Summary Plot - Class: approach")
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.close()

