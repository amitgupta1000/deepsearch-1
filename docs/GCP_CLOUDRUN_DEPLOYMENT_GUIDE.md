# Google Cloud Run Deployment Guide for INTELLISEARCH

This guide covers deploying INTELLISEARCH directly to Google Cloud Run from your repository, without requiring Docker containers.

## 📋 Prerequisites

1. **Google Cloud Project**
   - Active GCP project with billing enabled
   - Owner or Editor permissions

2. **Google Cloud SDK**
   ```bash
   # Install Google Cloud SDK
   # Visit: https://cloud.google.com/sdk/docs/install
   
   # Authenticate
   gcloud auth login
   
   # Set your project (gen-lang-client-0665888431)
   gcloud config set project gen-lang-client-0665888431
   ```

3. **API Keys**
   - Google AI API key (Gemini)
   - Serper API key (for search)

## 🚀 Quick Deployment

### Option 1: Automated Deployment (Recommended)

**For your project (gen-lang-client-0665888431):**

**Windows:**
```cmd
deploy-gen-lang-client.bat
```

**Linux/Mac:**
```bash
chmod +x deploy-gen-lang-client.sh
./deploy-gen-lang-client.sh
```

**Generic (any project):**
**Windows:**
```cmd
deploy-cloudrun.bat YOUR_PROJECT_ID
```

**Linux/Mac:**
```bash
chmod +x deploy-cloudrun.sh
./deploy-cloudrun.sh YOUR_PROJECT_ID
```

### Option 2: Manual Deployment

1. **Enable Required APIs**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   gcloud services enable secretmanager.googleapis.com
   ```

2. **Create Secrets**
   ```bash
   # Create secrets for API keys
   echo 'YOUR_GOOGLE_API_KEY' | gcloud secrets create google-api-key --data-file=-
   echo 'YOUR_SERPER_API_KEY' | gcloud secrets create serper-api-key --data-file=-
   ```

3. **Build and Deploy**
   ```bash
   # Build with Cloud Build
   gcloud builds submit --config cloudbuild.yaml
   
   # Deploy to Cloud Run
   gcloud run deploy intellisearch \
     --image gcr.io/YOUR_PROJECT_ID/intellisearch:latest \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --timeout 3600 \
     --max-instances 10 \
     --set-env-vars ENVIRONMENT=production,PRODUCTION_MODE=true
   ```

## 📁 Files Created for Cloud Run

### Core Files
- `requirements-cloudrun.txt` - Python dependencies optimized for Cloud Run
- `Dockerfile-cloudrun` - Multi-stage Docker build for Cloud Run
- `app_cloudrun.py` - FastAPI server with health checks and API endpoints
- `cloudbuild.yaml` - Cloud Build configuration for automated deployment
- `cloudrun-service.yaml` - Service configuration with resource limits

### Deployment Scripts
- `deploy-cloudrun.sh` - Linux/Mac deployment script
- `deploy-cloudrun.bat` - Windows deployment script

## 🔧 Configuration

### Environment Variables
The following environment variables are automatically set:

```yaml
# Core Application
PORT: 8080
ENVIRONMENT: production
PRODUCTION_MODE: true
PYTHONPATH: /app/src

# Performance Settings
LLM_TEMPERATURE: 0.1
MAX_TOKENS: 30000
MAX_SEARCH_QUERIES: 8
MAX_SEARCH_RESULTS: 8
MAX_CONCURRENT_SCRAPES: 4
CACHE_ENABLED: true
USE_ENHANCED_EMBEDDINGS: true
USE_HYBRID_RETRIEVAL: true
```

### Secrets (via Secret Manager)
- `GOOGLE_API_KEY` - Your Google AI API key
- `SERPER_API_KEY` - Your Serper search API key

## 🌐 API Endpoints

Once deployed, your service will have these endpoints:

### Health & Status
- `GET /health` - Health check endpoint
- `GET /ready` - Readiness check endpoint
- `GET /` - Root endpoint with API information

### Research APIs
- `POST /research` - Start async research task
- `POST /research/direct` - Run research and wait for completion
- `GET /task/{task_id}` - Get task status
- `GET /tasks` - List all tasks

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 🔍 API Usage Examples

### Start Research Task
```bash
curl -X POST "https://your-service-url/research" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Latest developments in quantum computing",
       "prompt_type": "general",
       "enable_automation": true
     }'
```

### Direct Research (Wait for completion)
```bash
curl -X POST "https://your-service-url/research/direct" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "AI ethics guidelines 2024",
       "prompt_type": "general"
     }'
```

### Check Task Status
```bash
curl "https://your-service-url/task/task_20241106_143022_0"
```

## 📊 Resource Configuration

### Cloud Run Settings
- **Memory**: 4 GB
- **CPU**: 2 vCPUs
- **Timeout**: 3600 seconds (1 hour)
- **Concurrency**: 100 requests per instance
- **Max Instances**: 10
- **Min Instances**: 0 (scales to zero)

### Performance Optimizations
- Gen2 execution environment
- CPU boost enabled
- Optimized startup probe
- Health checks configured
- Non-root user for security

## 🔐 Security Features

### Service Account
- Dedicated service account: `intellisearch-service-account`
- Minimal permissions (Secret Manager access only)
- No default compute engine service account usage

### Secrets Management
- API keys stored in Google Secret Manager
- Automatic secret rotation support
- No secrets in environment variables or code

### Container Security
- Non-root user execution
- Read-only root filesystem where possible
- Minimal base image (Python 3.10 slim)

## 📈 Monitoring & Logging

### Cloud Logging
```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=intellisearch" --limit=50

# Follow logs in real-time
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=intellisearch"
```

### Metrics
- Request latency
- Request count
- Error rate
- Memory and CPU usage
- Container startup time

## 🛠️ Development & Testing

### Local Testing
```bash
# Install dependencies
pip install -r requirements-cloudrun.txt

# Run locally
python app_cloudrun.py
```

### Environment Setup
Create a `.env` file for local development:
```env
GOOGLE_API_KEY=your_google_api_key
SERPER_API_KEY=your_serper_api_key
ENVIRONMENT=development
DEBUG_MODE=true
```

## 🔄 CI/CD Integration

### GitHub Actions (Example)
```yaml
name: Deploy to Cloud Run
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - id: 'auth'
      uses: 'google-github-actions/auth@v1'
      with:
        credentials_json: '${{ secrets.GCP_SA_KEY }}'
    - name: 'Deploy to Cloud Run'
      run: gcloud builds submit --config cloudbuild.yaml
```

## 🐛 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check build logs
   gcloud builds log [BUILD_ID]
   ```

2. **Service Not Starting**
   ```bash
   # Check service logs
   gcloud logging read "resource.type=cloud_run_revision" --limit=10
   ```

3. **API Key Issues**
   ```bash
   # Verify secrets
   gcloud secrets versions list google-api-key
   gcloud secrets versions list serper-api-key
   ```

4. **Memory/Timeout Issues**
   ```bash
   # Increase resources
   gcloud run services update intellisearch \
     --memory 8Gi \
     --timeout 3600 \
     --region us-central1
   ```

### Debug Mode
Enable debug logging by updating environment variables:
```bash
gcloud run services update intellisearch \
  --set-env-vars DEBUG_MODE=true,LOG_LEVEL=DEBUG \
  --region us-central1
```

## 💰 Cost Optimization

### Pricing Considerations
- Cloud Run pricing: CPU + Memory + Requests
- Scales to zero when not in use
- Pay only for actual usage
- Free tier: 2 million requests/month

### Optimization Tips
1. **Enable caching** to reduce API calls
2. **Optimize concurrency** for your workload
3. **Monitor cold starts** and adjust min instances if needed
4. **Use appropriate CPU/memory** for your use case

## 🔄 Updates & Maintenance

### Deploy Updates
```bash
# Redeploy latest code
gcloud builds submit --config cloudbuild.yaml

# Or deploy specific image
gcloud run deploy intellisearch \
  --image gcr.io/YOUR_PROJECT_ID/intellisearch:latest \
  --region us-central1
```

### Update Configuration
```bash
# Update environment variables
gcloud run services update intellisearch \
  --set-env-vars NEW_VAR=value \
  --region us-central1

# Update secrets
echo 'NEW_API_KEY' | gcloud secrets versions add google-api-key --data-file=-
```

This setup provides a production-ready, scalable deployment of INTELLISEARCH on Google Cloud Run with proper security, monitoring, and performance optimization.