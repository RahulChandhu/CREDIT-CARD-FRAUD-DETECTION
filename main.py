# main.py
"""
Streamlit app for credit card fraud detection.
Features:
- Upload CSV to make batch predictions
- Single-record form for manual prediction
- Shows sample predictions, explanation and model metrics if available
"""

import os
import pandas as pd
import streamlit as st
import joblib
from prediction_helper import load_model, prepare_single_input, predict_batch, preprocess, feature_engineer

MODEL_PATH = "best_model.joblib"

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

st.title("Credit Card Fraud Detection 🛡️")

# Load model if exists
model_loaded = None
selected_features = None
if os.path.exists(MODEL_PATH):
    try:
        model_loaded = joblib.load(MODEL_PATH)
        model, selected_features = model_loaded['model'], model_loaded['selected_features']
        st.sidebar.success(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
else:
    st.sidebar.warning("Model not found. Train and save model to 'models/best_model.joblib' first.")

st.sidebar.header("Options")
show_sample = st.sidebar.checkbox("Show sample data preview", value=True)
allow_download = st.sidebar.checkbox("Allow download of predictions", value=True)

# --------------
# Batch prediction via file upload
# --------------
st.header("Batch prediction (CSV upload)")
uploaded_file = st.file_uploader("Upload a CSV file with transactions (like creditcard.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    if show_sample:
        st.dataframe(df.head())

    if model_loaded is None:
        st.warning("No saved model available. Cannot predict. Train model first.")
    else:
        # Preprocess & feature engineering same as training pipeline
        df_proc = preprocess(df)
        df_proc = feature_engineer(df_proc)

        # Run prediction
        results = predict_batch(df_proc, model_obj=model_loaded)
        st.success("Predictions complete.")
        st.dataframe(results.head(50))

        if allow_download:
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")

# --------------
# Single-record prediction
# --------------
st.header("Single transaction prediction")
st.markdown("Enter features of a single transaction (only provide features used in model).")

if model_loaded is None:
    st.info("Model not loaded. Upload `models/best_model.joblib` into the project folder.")
else:
    with st.form("single_pred_form"):
        # Build inputs based on selected_features if available
        if selected_features is not None:
            input_vals = {}
            cols = selected_features
            left, right = st.columns(2)
            for i, col in enumerate(cols):
                if i % 2 == 0:
                    input_vals[col] = left.text_input(col, value="0")
                else:
                    input_vals[col] = right.text_input(col, value="0")
        else:
            # fallback small set
            input_vals = {
                "Amount": st.text_input("Amount", "0"),
                "V1": st.text_input("V1", "0"),
                "V2": st.text_input("V2", "0"),
            }

        submit = st.form_submit_button("Predict")
        if submit:
            # convert to numeric where possible
            rec = {}
            for k, v in input_vals.items():
                try:
                    rec[k] = float(v)
                except:
                    rec[k] = 0.0
            X = prepare_single_input(rec, selected_features)
            model = model_loaded['model']
            pred = model.predict(X)[0]
            score = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

            st.write("Prediction (1 = fraud, 0 = not fraud):", int(pred))
            if score is not None:
                st.write(f"Fraud probability score: {score:.4f}")

# --------------
# Show instructions
# --------------
st.sidebar.header("How to use")
st.sidebar.info(
    """
1. Train the model locally with `python prediction_helper.py --data creditcard.csv --save models/best_model.joblib`
2. Start the app: `streamlit run main.py`
3. Upload a CSV or use the single-record form to predict.
"""
)

st.sidebar.markdown("**Notes:** The app expects the model file at `models/best_model.joblib`.")

