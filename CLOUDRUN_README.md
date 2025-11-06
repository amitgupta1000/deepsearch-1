# 🚀 INTELLISEARCH - Google Cloud Run Deployment

**Quick deployment for project: `gen-lang-client-0665888431`**

## ⚡ Quick Start

### 1. Prerequisites
- Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Get API keys:
  - [Google AI API Key](https://aistudio.google.com/app/apikey)
  - [Serper API Key](https://serper.dev/)

### 2. Authenticate & Setup
```bash
# Authenticate with Google Cloud
gcloud auth login

# Quick setup (interactive - will ask for API keys)
# Windows:
quick-setup-gen-lang-client.bat

# Linux/Mac:
chmod +x quick-setup-gen-lang-client.sh
./quick-setup-gen-lang-client.sh
```

### 3. Deploy
```bash
# Windows:
deploy-gen-lang-client.bat

# Linux/Mac:
./deploy-gen-lang-client.sh
```

### 4. Test Your Deployment
```bash
# Your service will be available at:
# https://intellisearch-[hash]-uc.a.run.app

# Test health check
curl https://your-service-url/health

# Test research API
curl -X POST "https://your-service-url/research/direct" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is artificial intelligence?", "prompt_type": "general"}'

# Run comprehensive tests
python test_cloudrun_deployment.py https://your-service-url
```

## 📁 Deployment Files

### Project-Specific Files (gen-lang-client-0665888431):
- `quick-setup-gen-lang-client.sh/.bat` - Interactive API key setup
- `deploy-gen-lang-client.sh/.bat` - One-click deployment
- `cloudrun-service.yaml` - Pre-configured with your project ID

### Generic Files (any project):
- `deploy-cloudrun.sh/.bat` - Generic deployment scripts
- `requirements-cloudrun.txt` - Cloud Run optimized dependencies
- `app_cloudrun.py` - FastAPI server with health checks
- `Dockerfile-cloudrun` - Optimized container build
- `cloudbuild.yaml` - Cloud Build configuration

### Testing & Documentation:
- `test_cloudrun_deployment.py` - Test deployed service
- `docs/GCP_CLOUDRUN_DEPLOYMENT_GUIDE.md` - Comprehensive guide
- `.env.example` - Environment configuration template

## 🌟 Features

- **Zero-downtime deployment** with health checks
- **Auto-scaling** (0-10 instances) - pay only for usage
- **Secure** - API keys in Secret Manager, non-root execution
- **Production-ready** - logging, monitoring, error handling
- **Fast startup** - optimized container with Python 3.10
- **Complete API** - async/sync research endpoints
- **Easy testing** - built-in test scripts and health checks

## 📚 API Endpoints

- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /research` - Start async research task
- `POST /research/direct` - Synchronous research
- `GET /task/{task_id}` - Check task status

## 💡 Need Help?

1. **Full Documentation**: [GCP_CLOUDRUN_DEPLOYMENT_GUIDE.md](docs/GCP_CLOUDRUN_DEPLOYMENT_GUIDE.md)
2. **View Logs**: `gcloud logging read "resource.type=cloud_run_revision" --project=gen-lang-client-0665888431 --limit=10`
3. **Check Service**: `gcloud run services list --project=gen-lang-client-0665888431`

---

**Your project ID**: `gen-lang-client-0665888431`  
**Default region**: `us-central1`  
**Service name**: `intellisearch`