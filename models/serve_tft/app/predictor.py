import flask
import pandas as pd
import joblib
import boto3
import os
import json
from io import StringIO
from darts import TimeSeries
from darts.models import TFTModel
import torch

# --- 1. Configuration ---
S3_BUCKET = 'refit-project' # Your S3 bucket name
SCALER_KEY = 'models/scaler.gz'
# Darts models save two files. We reference the main .pkl file.
MODEL_KEY = 'models/tft_model.pkl' 
# The .ckpt file is assumed to be in the same S3 path
CKPT_KEY = 'models/tft_model.pkl.ckpt' 
LOCAL_MODEL_DIR = '/opt/ml/model'

# --- 2. Load Model and Scaler at Startup ---
s3 = boto3.client('s3')
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
model = None
scaler = None
try:
    # Download and load scaler
    s3.download_file(S3_BUCKET, SCALER_KEY, os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    scaler = joblib.load(os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    print("Scaler loaded successfully.")

    # Download both TFT files
    s3.download_file(S3_BUCKET, MODEL_KEY, os.path.join(LOCAL_MODEL_DIR, 'tft_model.pkl'))
    s3.download_file(S3_BUCKET, CKPT_KEY, os.path.join(LOCAL_MODEL_DIR, 'tft_model.ckpt'))
    
    # Load the Darts model (it will find the .ckpt file automatically)
    model = TFTModel.load(os.path.join(LOCAL_MODEL_DIR, 'tft_model.pkl'),
                          map_location = torch.device('cpu'))
    print("TFT model loaded successfully.")
    
except Exception as e:
    print(f"Error loading model or scaler: {e}")

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    status = 200 if model and scaler else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    if not model:
        return flask.Response(response='Model not loaded', status=500)

    content_type = flask.request.content_type
    
    # 1. Parse Input Data (expecting application/json)
    # Payload format: {"n": 24, "data_csv": "Time,Aggregate,hour,..."}
    if content_type == 'application/json':
        input_json = flask.request.get_json()
        forecast_length = input_json['n'] # How many hours to predict
        csv_data = input_json['data_csv']
        
        # The input data CSV must contain all features for the
        # historical lookback period AND the future forecast period
        input_df = pd.read_csv(StringIO(csv_data))
        input_df['Time'] = pd.to_datetime(input_df['Time'])
        input_df.set_index('Time', inplace=True)
    else:
        return flask.Response(response=f'Unsupported content type: {content_type}', status=415)

    # 2. Prepare Darts TimeSeries objects
    try:
        # Define all features the model was trained on
        features = ['hour', 'dayofweek', 'month', 'is_weekend', 'lag_1hr', 'lag_24hr', 'lag_168hr']
        # Note: The 'Aggregate' column is also needed for the history
        
        # The 'series' is the historical target data the model predicts from
        # This is all data *except* the last 'n' steps
        history_df = input_df.iloc[:-forecast_length]
        history_series = TimeSeries.from_series(history_df['Aggregate'], freq='H')
        
        # The 'future_covariates' are the features for the entire period
        # (history + future)
        future_covs_series = TimeSeries.from_dataframe(input_df[features], freq='H')

    except Exception as e:
        return flask.Response(response=f'Error creating TimeSeries objects: {e}', status=400)

    # 3. Predict
    try:
        predictions = model.predict(
            n=forecast_length,
            series=history_series, # The history
            future_covariates=future_covs_series, # Features for history + future
            num_samples=100 # For probabilistic forecast
        )
        
        # Extract the quantile predictions
        result_df = predictions.quantile_df([0.05, 0.50, 0.95])
        
    except Exception as e:
        return flask.Response(response=f'Error during prediction: {e}', status=400)

    # 4. Format Output as JSON
    # Convert the DataFrame to a split-oriented JSON string
    result_json = result_df.to_json(orient='split')

    return flask.Response(response=result_json, status=200, mimetype='application/json')