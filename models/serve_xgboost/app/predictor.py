import flask
import pandas as pd
import joblib
import boto3
import os
import xgboost as xgb # <-- Import XGBoost
import json
from io import StringIO

# --- Configuration ---
S3_BUCKET = 'refit-project' # Your S3 bucket name
MODEL_KEY = 'models/xgboost_model.pkl'  # The path to this specific model
SCALER_KEY = 'models/scaler.gz'
LOCAL_MODEL_DIR = '/opt/ml/model'

# --- Load Model and Scaler at Startup ---
s3 = boto3.client('s3')
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
model = None
scaler = None
try:
    s3.download_file(S3_BUCKET, MODEL_KEY, os.path.join(LOCAL_MODEL_DIR, 'xgboost_model.pkl'))
    s3.download_file(S3_BUCKET, SCALER_KEY, os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    
    model = joblib.load(os.path.join(LOCAL_MODEL_DIR, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    print("XGBoost model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    # Health check
    status = 200 if model and scaler else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    if not model:
        return flask.Response(response='Model not loaded', status=500)

    content_type = flask.request.content_type
    
    # 1. Parse Input Data (expecting text/csv)
    if content_type == 'text/csv':
        csv_data = flask.request.data.decode('utf-8')
        input_df = pd.read_csv(StringIO(csv_data))
        # Add any necessary index/dtype conversions here if needed
    else:
        return flask.Response(response=f'Unsupported content type: {content_type}', status=415)

    # 2. Define Features and Predict
    try:
        features = ['hour', 'dayofweek', 'month', 'is_weekend', 'lag_1hr', 'lag_24hr', 'lag_168hr']
        X_predict = input_df[features]
        predictions = model.predict(X_predict)
    except Exception as e:
        return flask.Response(response=f'Error during prediction: {e}', status=400)

    # 3. Format Output as JSON
    result = {'predictions': predictions.tolist()}
    result_json = json.dumps(result)

    return flask.Response(response=result_json, status=200, mimetype='application/json')