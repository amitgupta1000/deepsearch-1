@echo off
REM Build script for INTELLISEARCH deployment testing (Windows)

echo 🚀 Building INTELLISEARCH for deployment...

REM Build frontend
echo 📦 Building frontend...
cd web-app\frontend
call npm install --legacy-peer-deps
call npm run build

if %errorlevel% neq 0 (
    echo ❌ Frontend build failed!
    exit /b 1
)

echo ✅ Frontend build successful!

REM Test backend
echo 🔧 Testing backend...
cd ..\backend
python -c "import main; print('✅ Backend imports successful!')"

if %errorlevel% neq 0 (
    echo ❌ Backend test failed!
    exit /b 1
)

echo ✅ Backend test successful!
echo 🎉 Build completed successfully!
echo.
echo Next steps:
echo 1. Push your code to GitHub
echo 2. Follow the RENDER_DEPLOYMENT_GUIDE.md
echo 3. Deploy to Render.com

pause