import json
import flask
import pandas as pd
import joblib
import boto3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json
from io import StringIO

# --- 1. Define Helper Functions ---
# Define the *combined* quantile loss function that the model was trained with.
# This is REQUIRED for Keras to load the model.
# This function assumes the model outputs 3 quantiles: 0.05, 0.50, 0.95
QUANTILES = [0.05, 0.50, 0.95]

def combined_quantile_loss(y_true, y_pred):
    total_loss = 0.0
    for i, q in enumerate(QUANTILES):
        # Assumes y_pred has shape (batch_size, 3)
        error = tf.subtract(y_true, y_pred[:, i]) # Get the i-th output column
        loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
        total_loss += loss
    return total_loss / len(QUANTILES) # Return the average loss

# This function converts the 2D input data into 3D sequences
def create_sequences(data, lookback):
    X = []
    # Start from `lookback` index so we have a full history
    for i in range(lookback, len(data) + 1):
        feature_window = data[i-lookback:i]
        X.append(feature_window)
    return np.array(X)

# --- 2. Configuration ---
S3_BUCKET = 'refit-project'
# Use the name of your single combined model
MODEL_KEY = 'models/lstm_model.keras' 
SCALER_KEY = 'models/scaler.gz'
LOCAL_MODEL_DIR = '/opt/ml/model'
LOOKBACK = 48 # This MUST match the LOOKBACK you trained with

model = None
scaler = None

# --- 3. Load Model and Scaler at Startup ---
s3 = boto3.client('s3')
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
try:
    # Download and load scaler
    s3.download_file(S3_BUCKET, SCALER_KEY, os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    scaler = joblib.load(os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    print("Scaler loaded successfully.")

    # Download and load the single combined model
    model_name = os.path.basename(MODEL_KEY)
    local_path = os.path.join(LOCAL_MODEL_DIR, model_name)
    s3.download_file(S3_BUCKET, MODEL_KEY, local_path)
    
    # Pass the custom *combined* loss function to load_model
    model = load_model(
        local_path,
        custom_objects={'combined_quantile_loss': combined_quantile_loss},
        safe_mode = False 
    )
    print(f"Combined LSTM model loaded successfully.")
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
    
    # 1. Parse Input Data (expecting text/csv)
    # IMPORTANT: The input CSV must contain the prediction period PLUS the 48-hour
    # lookback period before it.
    if content_type == 'text/csv':
        csv_data = flask.request.data.decode('utf-8')
        input_df = pd.read_csv(StringIO(csv_data))
    else:
        return flask.Response(response=f'Unsupported content type: {content_type}', status=415)

    # 2. Create 3D Sequences from the 2D Input
    try:
        # We need ALL features for the model
        features = ['Aggregate', 'hour', 'dayofweek', 'month', 'is_weekend', 'lag_1hr', 'lag_24hr', 'lag_168hr'] # Add all features
        # Ensure the feature order is exactly the same as in training
        input_features = input_df[features] 
        
        # Create sequences. This will create (n_samples, LOOKBACK, n_features)
        X_pred = create_sequences(input_features.values, LOOKBACK)
        
        if X_pred.shape[0] == 0:
            return flask.Response(response='Input data not long enough to create sequences.', status=400)

    except Exception as e:
        return flask.Response(response=f'Error creating sequences: {e}', status=400)

    # 3. Predict with the combined model
    # The model will output an array of shape (n_samples, 3)
    preds = model.predict(X_pred)

    # 4. Format Output as JSON
    # Separate the 3 output columns
    predictions = {
        'q05': preds[:, 0].flatten().tolist(), # 5th percentile
        'q50': preds[:, 1].flatten().tolist(), # 50th percentile (median)
        'q95': preds[:, 2].flatten().tolist()  # 95th percentile
    }
    result_json = json.dumps(predictions)

    return flask.Response(response=result_json, status=200, mimetype='application/json')