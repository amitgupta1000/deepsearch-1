@echo off
REM quick-setup-gen-lang-client.bat - Quick setup script for gen-lang-client-0665888431
REM This script helps you set up API keys quickly for your INTELLISEARCH deployment

set PROJECT_ID=gen-lang-client-0665888431

echo 🚀 Quick Setup for INTELLISEARCH - Project: %PROJECT_ID%
echo ================================================================

REM Set project
echo 📋 Setting GCP project...
gcloud config set project %PROJECT_ID%

REM Check authentication
gcloud auth list --filter=status:ACTIVE --format="value(account)" >nul 2>&1
if errorlevel 1 (
    echo ❌ You need to authenticate with Google Cloud first:
    echo    gcloud auth login
    exit /b 1
)

echo ✅ Authenticated and project set to: %PROJECT_ID%

REM Setup Google API Key
echo.
echo 🔑 Setting up Google AI API Key...
gcloud secrets describe google-api-key --project=%PROJECT_ID% >nul 2>&1
if errorlevel 1 (
    set /p "google_key=   Enter your Google AI API Key: "
    echo !google_key! | gcloud secrets create google-api-key --data-file=- --project=%PROJECT_ID%
    echo    ✅ Google AI API Key created successfully!
) else (
    echo    Secret 'google-api-key' already exists.
    set /p "update_google=   Do you want to update it? (y/n): "
    if /i "!update_google!"=="y" (
        set /p "google_key=   Enter your Google AI API Key: "
        echo !google_key! | gcloud secrets versions add google-api-key --data-file=- --project=%PROJECT_ID%
        echo    ✅ Google AI API Key updated successfully!
    ) else (
        echo    ⏭️  Skipping Google AI API Key update.
    )
)

REM Setup Serper API Key
echo.
echo 🔑 Setting up Serper API Key...
gcloud secrets describe serper-api-key --project=%PROJECT_ID% >nul 2>&1
if errorlevel 1 (
    set /p "serper_key=   Enter your Serper API Key: "
    echo !serper_key! | gcloud secrets create serper-api-key --data-file=- --project=%PROJECT_ID%
    echo    ✅ Serper API Key created successfully!
) else (
    echo    Secret 'serper-api-key' already exists.
    set /p "update_serper=   Do you want to update it? (y/n): "
    if /i "!update_serper!"=="y" (
        set /p "serper_key=   Enter your Serper API Key: "
        echo !serper_key! | gcloud secrets versions add serper-api-key --data-file=- --project=%PROJECT_ID%
        echo    ✅ Serper API Key updated successfully!
    ) else (
        echo    ⏭️  Skipping Serper API Key update.
    )
)

echo.
echo 🎉 Setup completed! You can now deploy with:
echo.
echo    deploy-gen-lang-client.bat
echo.
echo 📚 Or view the full guide: docs\GCP_CLOUDRUN_DEPLOYMENT_GUIDE.md