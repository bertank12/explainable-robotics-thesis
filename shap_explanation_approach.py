import pandas as pd
import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load patched datasets containing multiple human-robot interaction intents
data_files = [
    "panda_data_approach_patched.csv", "panda_data_collapse.csv",
    "panda_data_grasp_patched.csv", "panda_data_handshake_patched.csv",
    "panda_data_idle_patched.csv", "panda_data_pushaway_patched.csv",
    "panda_data_retreat_patched.csv", "panda_data_wave_patched.csv"
]

# Combine all CSV files into one DataFrame
df = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)

# Define features and target label
X = df[["x", "y", "z", "vx", "vy", "vz", "force"]]
y = df["label"]

# Encode label classes into numeric format
le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels = le.classes_

# Split data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Train XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Initialize SHAP explainer for model interpretation
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Generate explanation for a sample belonging to the 'approach' intent
target_class = "approach"
class_index = list(labels).index(target_class)

# Find first sample in test set predicted as target class
sample_idx = np.where(y_test == class_index)[0][0]
sample_shap = shap_values.values[sample_idx, class_index, :len(X.columns)]
sample_features = X_test.iloc[sample_idx]

# Identify top 3 most influential features
top_indices = np.argsort(np.abs(sample_shap))[::-1][:3]

# Generate textual explanation
explanation = f"SHAP-Based Explanation for Human-Robot Intent: '{target_class}'\n\n"
explanation += f"The following features had the most impact on the model's decision:\n\n"

for idx in top_indices:
    fname = X.columns[idx]
    fval = sample_features[fname]
    sval = sample_shap[idx]
    direction = "increased" if sval > 0 else "decreased"
    explanation += f"- Feature '{fname}' with a value of {fval:.3f} {direction} the confidence for predicting '{target_class}'.\n"

# Save explanation to a text file
with open("shap_explanation_approach.txt", "w") as f:
    f.write(explanation)

print("SHAP explanation successfully saved to shap_explanation_approach.txt")

