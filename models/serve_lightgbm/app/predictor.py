import json
import flask
import pandas as pd
import joblib
import boto3
import os
import lightgbm as lgb # Specific import for this model

# --- Configuration ---
S3_BUCKET = 'refit-project'
MODEL_KEY = 'models/lightgbm_model.pkl' # Specific model file
SCALER_KEY = 'models/scaler.gz'
LOCAL_MODEL_DIR = '/opt/ml/model'

# --- Load Model and Scaler at Startup ---
s3 = boto3.client('s3')
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
model = None
scaler = None
try:
    s3.download_file(S3_BUCKET, MODEL_KEY, os.path.join(LOCAL_MODEL_DIR, 'lightgbm_model.pkl'))
    s3.download_file(S3_BUCKET, SCALER_KEY, os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    model = joblib.load(os.path.join(LOCAL_MODEL_DIR, 'lightgbm_model.pkl'))
    scaler = joblib.load(os.path.join(LOCAL_MODEL_DIR, 'scaler.gz'))
    print("LightGBM model and scaler loaded.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    status = 200 if model and scaler else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])

def transformation():
    if not model: return flask.Response(response='Model not loaded', status=500)

    content_type = flask.request.content_type
    
    # --- THIS IS THE UPDATED LOGIC ---
    if content_type == 'text/csv':
        from io import StringIO
        csv_data = flask.request.data.decode('utf-8')
        input_df = pd.read_csv(StringIO(csv_data))
        # Add any necessary index/dtype conversions here
        
    elif content_type == 'application/json':
        input_json = flask.request.get_json()
        input_df = pd.read_json(input_json['data'], orient='split')
        
    else:
        return flask.Response(response=f'Unsupported content type: {content_type}', status=415)
    # --- END OF UPDATED LOGIC ---

    # --- Prediction logic (remains the same) ---
    features = ['hour', 'dayofweek', 'month', 'is_weekend', 'lag_1hr', 'lag_24hr', 'lag_168hr']
    X_predict = input_df[features]
    predictions = model.predict(X_predict)

    # --- Response logic (remains the same, returns JSON) ---
    result = {'predictions': predictions.tolist()}
    result_json = json.dumps(result)
    return flask.Response(response=result_json, status=200, mimetype='application/json')

# Optional: Add wsgi.py for gunicorn if needed
# import predictor
# app = predictor.app