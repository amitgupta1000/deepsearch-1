#!/bin/bash
# deploy-cloudrun.sh - Deployment script for Google Cloud Run
# Make sure you're authenticated: gcloud auth login
# Set your project: gcloud config set project YOUR_PROJECT_ID

set -e

# Configuration
PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-"us-central1"}
SERVICE_NAME="intellisearch"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying INTELLISEARCH to Google Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID not set. Either pass it as first argument or set with 'gcloud config set project YOUR_PROJECT_ID'"
    exit 1
fi

# Ensure required APIs are enabled
echo "📋 Checking required APIs..."
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID

# Create secrets if they don't exist (you'll need to add the actual values)
echo "🔐 Setting up secrets..."

# Check if secrets exist, create if they don't
if ! gcloud secrets describe google-api-key --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "Creating google-api-key secret (you'll need to add the actual value later)"
    echo "placeholder" | gcloud secrets create google-api-key --data-file=- --project=$PROJECT_ID
    echo "⚠️  Remember to update the google-api-key secret with your actual Google API key:"
    echo "   echo 'YOUR_GOOGLE_API_KEY' | gcloud secrets versions add google-api-key --data-file=-"
fi

if ! gcloud secrets describe serper-api-key --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "Creating serper-api-key secret (you'll need to add the actual value later)"
    echo "placeholder" | gcloud secrets create serper-api-key --data-file=- --project=$PROJECT_ID
    echo "⚠️  Remember to update the serper-api-key secret with your actual Serper API key:"
    echo "   echo 'YOUR_SERPER_API_KEY' | gcloud secrets versions add serper-api-key --data-file=-"
fi

# Create service account if it doesn't exist
SERVICE_ACCOUNT="intellisearch-service-account@$PROJECT_ID.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "Creating service account..."
    gcloud iam service-accounts create intellisearch-service-account \
        --display-name="INTELLISEARCH Service Account" \
        --description="Service account for INTELLISEARCH Cloud Run service" \
        --project=$PROJECT_ID
    
    # Grant necessary permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="roles/secretmanager.secretAccessor"
fi

# Build and deploy using Cloud Build
echo "🏗️  Building and deploying with Cloud Build..."
gcloud builds submit --config cloudbuild.yaml --project=$PROJECT_ID

# Update service configuration with secrets
echo "🔧 Updating service configuration..."
gcloud run services update $SERVICE_NAME \
    --region=$REGION \
    --project=$PROJECT_ID \
    --update-secrets=GOOGLE_API_KEY=google-api-key:latest \
    --update-secrets=SERPER_API_KEY=serper-api-key:latest \
    --service-account=$SERVICE_ACCOUNT

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format="value(status.url)")

echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo "📚 API Documentation: $SERVICE_URL/docs"
echo "❤️  Health Check: $SERVICE_URL/health"

echo ""
echo "📋 Next steps:"
echo "1. Update your API keys in Secret Manager:"
echo "   echo 'YOUR_GOOGLE_API_KEY' | gcloud secrets versions add google-api-key --data-file=- --project=$PROJECT_ID"
echo "   echo 'YOUR_SERPER_API_KEY' | gcloud secrets versions add serper-api-key --data-file=- --project=$PROJECT_ID"
echo ""
echo "2. Test your service:"
echo "   curl $SERVICE_URL/health"
echo ""
echo "3. View logs:"
echo "   gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --project=$PROJECT_ID --limit=50"