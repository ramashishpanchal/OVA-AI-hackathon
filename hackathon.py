import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# -----------------------
# 1. Load Model and Encoders
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("hackathon.keras")

@st.cache_resource
def load_preprocessors():
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")  # dictionary of label encoders
    return scaler, encoders

model = load_model()
scaler, encoders = load_preprocessors()

# -----------------------
# 2. Streamlit UI
# -----------------------
st.title("🔍 Multi-Output Prediction App")
st.write("This app predicts **binary output (label)** and **multiclass output (attack category)** from uploaded data.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    # -----------------------
    # 3. Preprocessing
    # -----------------------
    st.write("### Preprocessing Data...")

    # Label Encoding for categorical columns
    for col, encoder in encoders.items():
        data[col] = data[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

    # Scaling
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    st.success("✅ Preprocessing Done!")

    # -----------------------
    # 4. Prediction
    # -----------------------
    st.write("### Predictions")
    predictions = model.predict(data_scaled)

    binary_preds = (predictions[0] > 0.5).astype(int).flatten()
    multiclass_preds = np.argmax(predictions[1], axis=1)

    result_df = data.copy()
    result_df["Predicted_Label"] = binary_preds
    result_df["Predicted_AttackCat"] = multiclass_preds

    st.dataframe(result_df.head())

    # Download option
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Please upload a CSV file to get started.")
