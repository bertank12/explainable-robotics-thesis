import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# TRAIN SET: original
original_files = [
    "panda_data_approach.csv", "panda_data_collapse.csv", "panda_data_grasp.csv",
    "panda_data_handshake.csv", "panda_data_idle.csv", "panda_data_pushaway.csv",
    "panda_data_retreat.csv", "panda_data_wave.csv"
]

# TEST SET: noisy
noisy_files = [
    "panda_data_approach_noisy.csv", "panda_data_collapse_noisy.csv", "panda_data_grasp_noisy.csv",
    "panda_data_handshake_noisy.csv", "panda_data_idle_noisy.csv", "panda_data_pushaway_noisy.csv",
    "panda_data_retreat_noisy.csv", "panda_data_wave_noisy.csv"
]

# Load and merge
df_train = pd.concat([pd.read_csv(f) for f in original_files], ignore_index=True)
df_test = pd.concat([pd.read_csv(f) for f in noisy_files], ignore_index=True)

X_train = df_train[["x", "y", "z", "vx", "vy", "vz", "force"]]
y_train = df_train["label"]

X_test = df_test[["x", "y", "z", "vx", "vy", "vz", "force"]]
y_test = df_test["label"]

# Label encoding
labels = sorted(y_train.unique().tolist())
label_map = {label: idx for idx, label in enumerate(labels)}
y_train_encoded = y_train.map(label_map)
y_test_encoded = y_test.map(label_map)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train_encoded)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test_encoded)
print("Noisy Test Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=labels))

