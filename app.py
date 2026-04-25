import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AI Threat Detection Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/best_model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    features = joblib.load("artifacts/feature_names.pkl")
    try:
        cat_info = joblib.load("artifacts/categorical_info.pkl")
    except FileNotFoundError:
        cat_info = {}
    return model, scaler, features, cat_info


try:
    model, scaler, features, cat_info = load_artifacts()
except FileNotFoundError as error:
    st.error(f"❌ Could not load model files: {error}")
    st.info(
        "Make sure artifacts/best_model.pkl, artifacts/scaler.pkl, and artifacts/feature_names.pkl exist"
    )
    st.stop()


if "log" not in st.session_state:
    st.session_state.log = []


st.title("🛡️ AI-Based Threat Detection Tool")
st.caption("ITM-390: Machine Learning | American University of Phnom Penh")
st.write(
    "Enter network activity features below to classify the traffic as Normal or "
    "a Threat using a trained Machine Learning model."
)
st.divider()


with st.sidebar:
    st.header("⚙️ Model Information")
    st.write("**Model:** Random Forest")
    st.write(f"**Number of features:** {len(features)}")

    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": ["97.98%", "98.36%", "97.28%", "97.82%"],
    }
    st.table(pd.DataFrame(metrics))
    st.success("✅ Model loaded and ready")


st.subheader("Network Feature Inputs")
cols = st.columns(3)
input_values = {}


def parse_numeric_input(raw_value: str) -> float:
    value = raw_value.strip()
    if value == "":
        return 0.0
    return float(value)

for index, feature in enumerate(features):
    with cols[index % 3]:
        label_text = feature.replace("_", " ").title()
        if feature in cat_info:
            options = cat_info[feature]
            val = st.selectbox(label=label_text, options=options, key=feature)
            input_values[feature] = val
        else:
            raw_value = st.text_input(label=label_text, value="0", key=feature)
            try:
                input_values[feature] = parse_numeric_input(raw_value)
            except ValueError:
                st.error(f"Invalid number for {feature}. Using 0.")
                input_values[feature] = 0.0

analyze = st.button("🔍 Analyze Traffic", width="stretch", type="primary")

if analyze:
    input_df = pd.DataFrame([input_values])[features]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    threat_prob = float(probability[1])
    normal_prob = float(probability[0])

    st.session_state.log.append(
        {
            "Result": "🔴 THREAT" if int(prediction) == 1 else "🟢 NORMAL",
            "Threat %": f"{threat_prob:.2%}",
            "Normal %": f"{normal_prob:.2%}",
            "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        }
    )

    st.divider()
    st.subheader("🔎 Detection Result")

    result_col, prob_col = st.columns([1, 2])

    with result_col:
        if int(prediction) == 1:
            st.error("## 🔴 THREAT DETECTED")
            st.markdown("This network activity has been classified as **malicious**.")
        else:
            st.success("## 🟢 NORMAL TRAFFIC")
            st.markdown("This network activity appears to be **safe**.")

    with prob_col:
        st.markdown("**Prediction Confidence**")
        st.progress(threat_prob, text=f"Threat Probability: {threat_prob:.2%}")
        st.progress(normal_prob, text=f"Normal Probability: {normal_prob:.2%}")

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Normal", "Threat"], [normal_prob, threat_prob], color=["#2ecc71", "#e74c3c"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Confidence Breakdown")
        st.pyplot(fig)
        plt.close(fig)

if st.session_state.log:
    st.divider()
    st.subheader("📋 Session Prediction Log")
    log_df = pd.DataFrame(st.session_state.log[::-1])
    st.dataframe(log_df, width="stretch")

    if st.button("🗑️ Clear Log"):
        st.session_state.log = []
        st.rerun()

st.divider()
st.caption(
    "ITM-390: Machine Learning | American University of Phnom Penh | "
    "School of Digital Technologies | AI-Based Threat Detection Tool"
)
