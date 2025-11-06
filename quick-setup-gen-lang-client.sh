#!/bin/bash
# quick-setup-gen-lang-client.sh - Quick setup script for gen-lang-client-0665888431
# This script helps you set up API keys quickly for your INTELLISEARCH deployment

PROJECT_ID="gen-lang-client-0665888431"

echo "🚀 Quick Setup for INTELLISEARCH - Project: $PROJECT_ID"
echo "================================================================"

# Set project
echo "📋 Setting GCP project..."
gcloud config set project $PROJECT_ID

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 >/dev/null 2>&1; then
    echo "❌ You need to authenticate with Google Cloud first:"
    echo "   gcloud auth login"
    exit 1
fi

echo "✅ Authenticated and project set to: $PROJECT_ID"

# Function to update API key
update_api_key() {
    local key_name=$1
    local key_description=$2
    
    echo ""
    echo "🔑 Setting up $key_description..."
    
    # Check if secret exists
    if gcloud secrets describe $key_name --project=$PROJECT_ID >/dev/null 2>&1; then
        echo "   Secret '$key_name' already exists."
        read -p "   Do you want to update it? (y/n): " update_choice
        if [ "$update_choice" = "y" ] || [ "$update_choice" = "Y" ]; then
            read -s -p "   Enter your $key_description: " api_key
            echo ""
            echo "$api_key" | gcloud secrets versions add $key_name --data-file=- --project=$PROJECT_ID
            echo "   ✅ $key_description updated successfully!"
        else
            echo "   ⏭️  Skipping $key_description update."
        fi
    else
        read -s -p "   Enter your $key_description: " api_key
        echo ""
        echo "$api_key" | gcloud secrets create $key_name --data-file=- --project=$PROJECT_ID
        echo "   ✅ $key_description created successfully!"
    fi
}

# Setup API keys
update_api_key "google-api-key" "Google AI API Key"
update_api_key "serper-api-key" "Serper API Key"

echo ""
echo "🎉 Setup completed! You can now deploy with:"
echo ""
echo "   For Windows: deploy-gen-lang-client.bat"
echo "   For Linux/Mac: ./deploy-gen-lang-client.sh"
echo ""
echo "📚 Or view the full guide: docs/GCP_CLOUDRUN_DEPLOYMENT_GUIDE.md"