import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- DATA LOAD ---
files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# --- FEATURES / LABEL ---
X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- GRID SEARCH ---
param_grid = {
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100],
    "learning_rate": [0.05, 0.1]
}
clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", tree_method="hist")
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# --- BEST MODEL ---
best_model = grid_search.best_estimator_
print("Best Params:", grid_search.best_params_)

# --- PREDICTION ---
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix after GridSearchCV\nAccuracy: {acc:.4f}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_cv.png")
plt.show()

