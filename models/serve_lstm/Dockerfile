# Start with a Python base image matching your training environment
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app code
COPY ./app /app

# Environment variables SageMaker uses
ENV MODEL_DIR=/opt/ml/model
ENV SAGEMAKER_PROGRAM=predictor.py

# Expose the port Gunicorn will run on
EXPOSE 8080

# Keep the container running and start the Gunicorn web server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "predictor:app"]