# 1. Base Image
FROM python:3.9-slim

# 2. Set Working Directory
WORKDIR /app

# 3. Copy Requirements & Install
COPY requirements.txt .
# Install dependencies, including system libs needed by libraries like Scikit-learn, Numpy, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy Source Code, models, and data access files
COPY app/ .

# 5. Expose Streamlit Port
EXPOSE 8501

# 6. Run the Application
# Binds to 0.0.0.0 so it's accessible by App Runner
ENTRYPOINT python -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0