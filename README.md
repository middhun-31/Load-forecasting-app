### ⚡️Probabilistic Household Load Forecasting App

Tech Stack: Python 3.9+ • Streamlit • AWS App Runner • Docker • PyTorch • LightGBM • XGBoost

##🚀 Launch Live Demo

## Project Context: Uncertainty Quantification

1. Deployment layer for research on uncertainty quantification in energy data
2. Moves beyond point forecasts to provide calibrated uncertainty intervals
3. Comparative study of SARIMA vs. Gradient Boosting vs. Transformers
4.Research Repo: 

##Frontend Layer: Streamlit Dashboard
1. Containerized application running on AWS App Runner
2. Handles user inputs for House Selection and Date Range
3. Performs real-time data loading from Amazon S3
4. Implements custom Seasonal Imputation for preprocessing
5. Visualizes dynamic forecasts and 90% prediction intervals

## Backend Layer: Inference Services
1. Microservices architecture with four independent containers
2. Models: LightGBM, XGBoost, Probabilistic LSTM, Temporal Fusion Transformer (TFT)
3. REST API communication between frontend and inference engines
4. Real-time calculation of RMSE, MAE, and Coverage metrics

## Data Layer: Storage & Assets
1. Amazon S3: Secure storage for raw REFIT dataset and model artifacts
2. Multi-House Data: Supports generalization testing across 20 distinct households
3. Serverless: Cost-efficient architecture (Note: 10-15s cold start time)

33Local Development Setup

1. Clone the Repository
<pre>
git clone https://github.com/middhun-31/Load-forecasting-app.git
cd refit-streamlit-app
</pre>

2. Install Dependencies
<pre>
pip install -r requirements.txt
</pre>

3. Configure AWS Access
<pre>
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=eu-north-1
</pre>

4. Run the App
<pre>
streamlit run app.py
</pre>

(Hosts at localhost:8501)

Deployment Pipeline

1. Build the Docker Image
<pre>
docker build -t refit-streamlit-app .
</pre>

2. Push to ECR
<pre>
# Login to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com

# Tag and Push
docker tag refit-streamlit-app:latest <repo_uri>:latest
docker push <repo_uri>:latest
</pre>
