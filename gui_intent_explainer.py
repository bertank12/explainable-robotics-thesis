import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random

st.set_page_config(layout="wide")
st.title("Explainable Intent Recognition in Human-Robot Interaction")

# SHAP explanation - detailed text
def generate_explanation(sample, sample_shap, feature_names):
    explanations = []
    for fname, fval, sval in zip(feature_names, sample, sample_shap):
        sval_scalar = float(sval[0]) if isinstance(sval, (np.ndarray, list)) else float(sval)
        if abs(sval_scalar) < 0.05:
            continue
        direction = "increased" if fval > 0 else "decreased"
        impact = "positively contributed" if sval_scalar > 0 else "negatively influenced"
        explanations.append(f"- Feature **{fname}** {direction}, which {impact} the prediction.")
    if not explanations:
        return "No significant feature contributions were detected for this prediction."
    return "### SHAP-Based Explanation\n\n" + "\n".join(explanations)

# SHAP explanation - short summary
def generate_natural_text(sample, shap_values, feature_names, predicted_label):
    shap_scores = [
        (fname, fval, sval)
        for fname, fval, sval in zip(feature_names, sample, shap_values)
        if abs(sval) > 0.05
    ]
    sorted_scores = sorted(shap_scores, key=lambda x: abs(x[2]), reverse=True)[:3]
    if not sorted_scores:
        return "Not enough feature impact to generate explanation."
    phrases = []
    for fname, fval, sval in sorted_scores:
        change = "increased" if fval > 0 else "decreased"
        phrases.append(f"{fname} {change}")
    reason_text = " and ".join(phrases)
    return f"**Explanation Summary:** Because {reason_text}, the system predicted **{predicted_label}**."

# Model robustness to noise
def noise_vs_accuracy(model, X_test, y_test, sigmas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], repeats=5):
    results = []
    for sigma in sigmas:
        accs = []
        for _ in range(repeats):
            noisy_X = X_test + np.random.normal(0, sigma, X_test.shape)
            preds = model.predict(noisy_X)
            acc = np.mean(preds == y_test)
            accs.append(acc)
        avg_acc = np.mean(accs)
        results.append((sigma, avg_acc))
    return pd.DataFrame(results, columns=["Noise Level (σ)", "Accuracy"])

# SHAP value robustness to noise
def shap_stability_analysis(model, explainer, X_test, feature_names, n_samples=20, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4]):
    np.random.seed(42)
    sampled_indices = np.random.choice(len(X_test), size=n_samples, replace=False)
    X_sampled = X_test.iloc[sampled_indices]
    stability_data = []
    for sigma in noise_levels:
        total_diff = 0
        for i in range(n_samples):
            original = X_sampled.iloc[i].values.reshape(1, -1)
            noisy = original + np.random.normal(0, sigma, size=original.shape)
            shap_original = explainer.shap_values(original)[0]
            shap_noisy = explainer.shap_values(noisy)[0]
            diff = np.abs(shap_original - shap_noisy).mean()
            total_diff += diff
        avg_diff = total_diff / n_samples
        stability_data.append((sigma, avg_diff))
    return pd.DataFrame(stability_data, columns=["Noise σ", "Mean |SHAP Change|"])

# Model training and SHAP explainer
@st.cache_resource
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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return X, y_encoded, labels, X_test, y_test, model, explainer, shap_values

# Load model
with st.spinner("Training the model and calculating SHAP values..."):
    X, y_encoded, labels, X_test, y_test, model, explainer, shap_values_list = load_and_train()


# UI Input mode
mode = st.radio("Select Input Mode:", ["Manual", "Live Simulation"], horizontal=True)

if mode == "Manual":
    selected_class = st.selectbox("Select an Intent Class", labels)
    class_index = list(labels).index(selected_class)
    indices = np.where(y_test == class_index)[0]
    random_idx = random.choice(indices)
    sample = X_test.iloc[random_idx]
    noise_option = "Clean Data"
else:
    st.subheader("Live Simulation Settings")
    col_radio, col_button = st.columns([3, 1])

    with col_radio:
        noise_option = st.radio(
            "Select Input Type:",
            ["Clean Data", "Add Gaussian Noise (σ = 0.3)", "External Test Data"],
            horizontal=True
        )

    with col_button:
        st.markdown("<div style='margin-top: 30px'></div>", unsafe_allow_html=True)
        if st.button("Generate Sample"):
            st.rerun()

    with st.expander("More Info"):
        st.markdown("""
        **Note:** The noise option is used to evaluate the system's robustness by adding Gaussian noise  
        to the input features. This is a test-only feature and does not affect the reported model performance.  
        In deployment, better data and training will reduce this sensitivity.  
        \n**Noise Formula:** `x_noisy = x + N(0, σ)` where σ = 0.3
        """)

    random_idx = random.randint(0, len(X_test) - 1)
    selected_class = labels[y_test[random_idx]]
    base_sample = X_test.iloc[random_idx]
    σ = 0.3
    sample = base_sample + np.random.normal(0, σ, size=base_sample.shape) if noise_option == "Add Gaussian Noise (σ = 0.3)" else base_sample

# Prediction & SHAP
sample_true = y_test[random_idx]
sample_pred = model.predict([sample])[0]
predicted_label = int(sample_pred)
sample_shap = shap_values_list[predicted_label][random_idx]
summary_shap = shap_values_list[predicted_label]

# Impedance values
IMPEDANCE_PARAMS = {
    "approach": (20, 10), "idle": (60, 50), "retreat": (150, 80),
    "grasp": (70, 40), "wave": (90, 60), "pushaway": (120, 70),
    "collapse": (30, 20), "handshake": (80, 45)
}
K_val, B_val = IMPEDANCE_PARAMS.get(selected_class, (50, 30))

# Layout: Left
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prediction Information")
    st.write(f"**True Class:** {labels[sample_true]}")
    st.write(f"**Predicted Class:** {labels[sample_pred]}")
    correct = sample_true == sample_pred
    color = "darkgreen" if correct else "darkred"
    symbol = "✔" if correct else "✖"
    st.markdown(f'<span style="color:{color}; font-weight:bold;">{symbol} Prediction Correct</span>', unsafe_allow_html=True)

    st.subheader("Explanation")
    tab1, tab2 = st.tabs(["SHAP Feature Table", "Natural Language"])
    with tab1:
        for fname, fval, sval in zip(X.columns, sample, sample_shap):
            sval_scalar = float(sval[0]) if isinstance(sval, (np.ndarray, list)) else float(sval)
            sign = "+" if sval_scalar >= 0 else "-"
            st.write(f"- {fname}: {fval:.4f} → SHAP: {sign}{abs(sval_scalar):.4f}")
    with tab2:
        st.markdown(generate_explanation(sample, sample_shap, X.columns))
        st.markdown("---")
        st.markdown(generate_natural_text(sample, sample_shap, X.columns, labels[sample_pred]))

# Layout: Right tabs
with col2:
    tab_titles = ["SHAP Summary", "SHAP Bars", "Motion", "Gauges"]
    if mode == "Live Simulation":
        tab_titles.append("Sample Input")
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        fig1, ax1 = plt.subplots()
        shap.summary_plot(summary_shap, X_test, feature_names=X.columns, show=False)
        st.pyplot(fig1)

    with tabs[1]:
        fig2, ax2 = plt.subplots()
        mean_shap = np.abs(summary_shap).mean(axis=0)
        order = np.argsort(mean_shap)[::-1]
        ax2.barh([X.columns[i] for i in order], mean_shap[order])
        ax2.set_xlabel("Mean |SHAP value|")
        ax2.set_title("Feature Importance (Bar)")
        st.pyplot(fig2)

    with tabs[2]:
        gif_path = f"loopanimation/{selected_class}.gif"
        try:
            with open(gif_path, "rb") as f:
                st.image(f.read(), caption=f"{selected_class.upper()} Motion", use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Motion preview not found for '{selected_class}'")

    with tabs[3]:
        col_k, col_b = st.columns(2)
        with col_k:
            st.metric("Stiffness (K)", K_val)
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number", value=K_val, title={'text': "Stiffness (K)"},
                gauge={'axis': {'range': [0, 200]}}
            )), use_container_width=True)
        with col_b:
            st.metric("Damping (B)", B_val)
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number", value=B_val, title={'text': "Damping (B)"},
                gauge={'axis': {'range': [0, 100]}}
            )), use_container_width=True)

    if mode == "Live Simulation":
        with tabs[-1]:
            st.subheader("Random Sample Input Features")
            st.dataframe(sample.to_frame().T)

# --- Extra Graphs ---
if mode == "Live Simulation" and noise_option == "Add Gaussian Noise (σ = 0.3)":
    with st.expander("Graphs: Robustness and SHAP Stability", expanded=False):
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            st.subheader("Noise vs Accuracy")
            df_robust = noise_vs_accuracy(model, X_test, y_test)
            st.line_chart(df_robust.set_index("Noise Level (σ)"))
            st.markdown("Higher stability in accuracy under noise indicates better generalization.")

        with col_graph2:
            st.subheader("SHAP Stability under Noise")
            df_stability = shap_stability_analysis(model, explainer, X_test, X.columns)
            st.line_chart(df_stability.set_index("Noise σ"))
            st.markdown("Lower change in SHAP values under noise indicates stronger explanation consistency.")

