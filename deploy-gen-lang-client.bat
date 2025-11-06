@echo off
REM deploy-gen-lang-client.bat - Project-specific Windows deployment script for INTELLISEARCH
REM Configured for project: gen-lang-client-0665888431

setlocal enabledelayedexpansion

REM Project Configuration
set PROJECT_ID=gen-lang-client-0665888431
set REGION=us-central1
set SERVICE_NAME=intellisearch
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo 🚀 Deploying INTELLISEARCH to Google Cloud Run
echo Project ID: %PROJECT_ID%
echo Region: %REGION%
echo Service Name: %SERVICE_NAME%

REM Set the project (in case it's not already set)
gcloud config set project %PROJECT_ID%

REM Ensure required APIs are enabled
echo 📋 Enabling required APIs...
gcloud services enable run.googleapis.com --project=%PROJECT_ID%
gcloud services enable cloudbuild.googleapis.com --project=%PROJECT_ID%
gcloud services enable containerregistry.googleapis.com --project=%PROJECT_ID%
gcloud services enable secretmanager.googleapis.com --project=%PROJECT_ID%

REM Create secrets if they don't exist
echo 🔐 Setting up secrets...

REM Check and create google-api-key secret
gcloud secrets describe google-api-key --project=%PROJECT_ID% >nul 2>&1
if errorlevel 1 (
    echo Creating google-api-key secret ^(you'll need to add the actual value later^)
    echo placeholder | gcloud secrets create google-api-key --data-file=- --project=%PROJECT_ID%
    echo ⚠️  Remember to update the google-api-key secret with your actual Google API key:
    echo    echo 'YOUR_GOOGLE_API_KEY' ^| gcloud secrets versions add google-api-key --data-file=- --project=%PROJECT_ID%
)

REM Check and create serper-api-key secret
gcloud secrets describe serper-api-key --project=%PROJECT_ID% >nul 2>&1
if errorlevel 1 (
    echo Creating serper-api-key secret ^(you'll need to add the actual value later^)
    echo placeholder | gcloud secrets create serper-api-key --data-file=- --project=%PROJECT_ID%
    echo ⚠️  Remember to update the serper-api-key secret with your actual Serper API key:
    echo    echo 'YOUR_SERPER_API_KEY' ^| gcloud secrets versions add serper-api-key --data-file=- --project=%PROJECT_ID%
)

REM Create service account if it doesn't exist
set SERVICE_ACCOUNT=intellisearch-service-account@%PROJECT_ID%.iam.gserviceaccount.com
gcloud iam service-accounts describe %SERVICE_ACCOUNT% --project=%PROJECT_ID% >nul 2>&1
if errorlevel 1 (
    echo Creating service account...
    gcloud iam service-accounts create intellisearch-service-account --display-name="INTELLISEARCH Service Account" --description="Service account for INTELLISEARCH Cloud Run service" --project=%PROJECT_ID%
    
    REM Grant necessary permissions
    gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT%" --role="roles/secretmanager.secretAccessor"
)

REM Build and deploy using Cloud Build
echo 🏗️  Building and deploying with Cloud Build...
gcloud builds submit --config cloudbuild.yaml --project=%PROJECT_ID%

REM Update service configuration with secrets
echo 🔧 Updating service configuration...
gcloud run services update %SERVICE_NAME% --region=%REGION% --project=%PROJECT_ID% --update-secrets=GOOGLE_API_KEY=google-api-key:latest --update-secrets=SERPER_API_KEY=serper-api-key:latest --service-account=%SERVICE_ACCOUNT%

REM Get service URL
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --region=%REGION% --project=%PROJECT_ID% --format="value(status.url)"') do set SERVICE_URL=%%i

echo ✅ Deployment completed successfully!
echo 🌐 Service URL: %SERVICE_URL%
echo 📚 API Documentation: %SERVICE_URL%/docs
echo ❤️  Health Check: %SERVICE_URL%/health

echo.
echo 📋 Next steps:
echo 1. Update your API keys in Secret Manager:
echo    echo 'YOUR_GOOGLE_API_KEY' ^| gcloud secrets versions add google-api-key --data-file=- --project=%PROJECT_ID%
echo    echo 'YOUR_SERPER_API_KEY' ^| gcloud secrets versions add serper-api-key --data-file=- --project=%PROJECT_ID%
echo.
echo 2. Test your service:
echo    curl %SERVICE_URL%/health
echo    python test_cloudrun_deployment.py %SERVICE_URL%
echo.
echo 3. View logs:
echo    gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=%SERVICE_NAME%" --project=%PROJECT_ID% --limit=50
echo.
echo 4. Quick research test:
echo    curl -X POST "%SERVICE_URL%/research/direct" \
echo         -H "Content-Type: application/json" \
echo         -d "{\"query\": \"What is artificial intelligence?\", \"prompt_type\": \"general\"}"

endlocal