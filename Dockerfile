# Dockerfile for GCP Cloud Run
FROM mcr.microsoft.com/playwright/python:v1.50.0-noble
WORKDIR /app
# Copy only requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps

# Ensure Python can find backend as a package for absolute imports
ENV PYTHONPATH=/app

# Copy application code, preserving the directory structure
COPY main.py ./
COPY backend/ ./backend/

# This layer will cache the downloaded model in the image
COPY backend/src/download_model.py backend/src/config.py ./backend/src/
RUN python -m backend.src.download_model

# This __init__.py file makes 'backend' a Python package
RUN touch backend/__init__.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]