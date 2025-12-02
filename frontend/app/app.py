# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import json
import matplotlib.pyplot as plt
import boto3
from io import StringIO
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocessing import preprocess_for_streamlit, inverse_scale_predictions # Import utilities

# --- Configuration & Model URLs ---
MODELS_DIR = "models"
S3_BUCKET_NAME = 'refit-project' # S3 bucket where house_X.csv files live

# --- IMPORTANT: Replace with the actual live App Runner URLs ---
MODEL_URLS = {
    "LightGBM": "https://qfpjj342wa.eu-west-1.awsapprunner.com/invocations",
    "XGBoost": "https://guptwubi4c.eu-west-1.awsapprunner.com/invocations",
    "LSTM (Probabilistic)": "https://uw9kdfywrp.eu-west-1.awsapprunner.com/invocations",
    "TFT (Probabilistic)": "https://kkpk3z7mmf.eu-west-1.awsapprunner.com/invocations"
}
# ------------------------------------------------------------------

# --- Caching Functions ---
@st.cache_resource
def load_scaler():
    # In a real app, this scaler would be downloaded from S3 as well
    # For this demo, assume it is bundled in the container for simplicity
    return joblib.load(os.path.join(MODELS_DIR, "scaler.gz"))

@st.cache_data
def load_raw_data_from_s3(house_number):
    s3_client = boto3.client('s3')
    s3_key = f'house_data/House_{house_number}.csv' 
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        return df
    except Exception as e:
        st.error(f"Error loading {s3_key} from S3: {e}")
        return pd.DataFrame()

# --- Prediction Function (Calls External API) ---
def get_prediction(model_name, house_data_processed, start_date, end_date):
    api_url = MODEL_URLS[model_name]
    
    # The API needs the features for the entire context period
    # Convert Data to JSON/CSV Payload
    payload_df = house_data_processed.reset_index().rename(columns={'index': 'Time'})

    # --- For Probabilistic Models (TFT/LSTM), use JSON payload ---
    if model_name in ["TFT (Probabilistic)", "LSTM (Probabilistic)"]:
        forecast_n = len(house_data_processed.loc[start_date:end_date])
        
        json_payload = {
            "n": forecast_n, 
            "data_csv": payload_df.to_csv(index=False)
        }
        
        response = requests.post(
            api_url, 
            json=json_payload, 
            headers={'Content-Type': 'application/json'}
        )
    else:
        # --- For Point Models (LGBM/XGBoost), send raw CSV ---
        csv_payload = payload_df.to_csv(index=False)
        response = requests.post(
            api_url, 
            data=csv_payload, 
            headers={'Content-Type': 'text/csv'}
        )
        
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error ({response.status_code}): {response.text}")
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("⚡️ Probabilistic Load Forecasting Demo")

scaler_fitted = load_scaler()

with st.sidebar:
    st.header("1. Select Data & Model")
    selected_house = st.selectbox(
        "Household:",
        options=[f"House {i}" for i in range(2, 21)],
        index=0
    )
    house_number = int(selected_house.split(" ")[1])

    selected_model = st.selectbox(
        "Forecasting Model:",
        options=list(MODEL_URLS.keys()),
        index=3 # Default to TFT
    )

    st.header("2. Prediction Range")
    # Set a default range for the last week of the data
    default_end = pd.to_datetime("2015-07-01") 
    default_start = default_end - pd.Timedelta(days=7) 

    date_range = st.date_input(
        "Select Test Date Range:",
        value=(default_start, default_end)
    )

    if st.button("Generate Forecast", type="primary"):
        st.session_state['run_forecast'] = True
    else:
        st.session_state['run_forecast'] = False

# --- Main App Logic ---
if st.session_state.get('run_forecast', False):
    start_date, end_date = date_range
    st.markdown("---")

    with st.spinner(f"Loading data and calling {selected_model} endpoint..."):
        # 1. Load Data
        raw_df = load_raw_data_from_s3(house_number)
        
        # 2. Preprocess Data and Get Context
        try:
            # df_full_processed contains all features and the necessary historical context
            df_full_processed, df_validation_slice = preprocess_for_streamlit(
                raw_df.copy(), scaler_fitted, start_date, end_date
            )
        except Exception as e:
            st.error(f"Preprocessing Error: {e}")
            st.stop()
            
        # 3. Call API
        api_result = get_prediction(selected_model, df_full_processed, start_date, end_date)

    if api_result:
        with st.container():
            # 4. Parse Results and Inverse Scale
            actuals_unscaled = inverse_scale_predictions(df_validation_slice['Aggregate'].values, scaler_fitted)
            
            # --- Handle Probabilistic vs Point Response ---
            if 'q50' in api_result or 'q05' in api_result:
                # Probabilistic Output (LSTM/TFT)
                median_scaled = np.array(api_result.get('q50', api_result.get('predictions', [])), dtype=np.float32)
                lower_scaled = np.array(api_result.get('q05', api_result.get('predictions', [])), dtype=np.float32)
                upper_scaled = np.array(api_result.get('q95', api_result.get('predictions', [])), dtype=np.float32)
                
                # Inverse Scale
                median_unscaled = inverse_scale_predictions(median_scaled, scaler_fitted)
                lower_unscaled = inverse_scale_predictions(lower_scaled, scaler_fitted)
                upper_unscaled = inverse_scale_predictions(upper_scaled, scaler_fitted)

                # 5. Calculate Metrics
                rmse = np.sqrt(mean_squared_error(actuals_unscaled, median_unscaled))
                mae = mean_absolute_error(actuals_unscaled, median_unscaled)
                coverage = np.mean((actuals_unscaled >= lower_unscaled) & (actuals_unscaled <= upper_unscaled)) * 100

                st.markdown(f"### Results for {selected_house}")
                colA, colB, colC = st.columns(3)
                colA.metric("RMSE (Median)", f"{rmse:.2f}")
                colB.metric("MAE (Median)", f"{mae:.2f}")
                colC.metric("90% Coverage", f"{coverage:.2f}%")
                
                # 6. Plot Probabilistic Forecast
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_validation_slice.index, actuals_unscaled, 'k', label='Actual Load')
                ax.plot(df_validation_slice.index, median_unscaled, 'b--', label='Median Forecast')
                ax.fill_between(df_validation_slice.index, lower_unscaled, upper_unscaled, color='blue', alpha=0.2, label='90% Prediction Interval')
                ax.set_title(f'{selected_model} Forecast for {start_date.strftime("%b %d")} - {end_date.strftime("%b %d")}')
                ax.set_xlabel("Time")
                ax.set_ylabel("Load (Original Units)")
                ax.legend()
                st.pyplot(fig)


            else:
                # Point Forecast Output (LGBM/XGB)
                preds_scaled = np.array(api_result['predictions'], dtype=np.float32)
                preds_unscaled = inverse_scale_predictions(preds_scaled, scaler_fitted)

                rmse = np.sqrt(mean_squared_error(actuals_unscaled, preds_unscaled))
                mae = mean_absolute_error(actuals_unscaled, preds_unscaled)

                st.markdown(f"### Results for {selected_house}")
                st.metric("RMSE", f"{rmse:.2f} (Watts)")
                st.metric("MAE", f"{mae:.2f} (Watts)")

                # Plot Point Forecast
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_validation_slice.index, actuals_unscaled, 'k', label='Actual Load')
                ax.plot(df_validation_slice.index, preds_unscaled, 'r--', label='Point Forecast')
                ax.set_title(f'{selected_model} Forecast for {start_date.strftime("%b %d")} - {end_date.strftime("%b %d")}')
                ax.set_xlabel("Time")
                ax.set_ylabel("Load (Original Units)")
                ax.legend()
                st.pyplot(fig)