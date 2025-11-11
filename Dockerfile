# Dockerfile for GCP Cloud Run
FROM python:3.11-slim
WORKDIR /app
# Copy only requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Copy only source code and necessary files
COPY backend/src/ ./src/
COPY main.py ./
COPY backend/app.py ./
COPY backend/auth.py ./
# If you need config files, copy them explicitly
# COPY config.yaml ./
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
