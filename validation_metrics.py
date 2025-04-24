import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veri ve model yükleme
def load_and_train():
    files = [
        "panda_data_approach_patched.csv", "panda_data_collapse.csv",
        "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
        "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
        "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", tree_method="hist")
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, labels

# Ana işlemler
model, X_train, X_test, y_train, y_test, labels = load_and_train()

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", np.round(cv_scores, 3))
print("Mean Accuracy:", np.mean(cv_scores))
print("Std Dev:", np.std(cv_scores))

# Confusion Matrix & Classification Report
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# Confusion matrix görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

