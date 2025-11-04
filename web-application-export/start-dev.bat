@echo off
REM Development startup script for INTELLISEARCH Web Application
REM This script starts both the backend and frontend in development mode

echo 🚀 Starting INTELLISEARCH Web Application...
echo.

REM Check prerequisites
echo Checking prerequisites...

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)
echo ✅ Python found

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    pause
    exit /b 1
)
echo ✅ Node.js found

REM Check npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not installed or not in PATH
    pause
    exit /b 1
)
echo ✅ npm found

echo.

REM Install dependencies if needed
echo Installing dependencies...

REM Backend dependencies
cd web-app\backend
pip install fastapi uvicorn python-multipart >nul 2>&1
cd ..\..

REM Frontend dependencies
cd web-app\frontend
if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install --legacy-peer-deps
)
cd ..\..

echo.
echo 🎉 Starting services...
echo.

REM Start backend
echo Starting FastAPI backend on http://localhost:8000
cd web-app\backend
start "INTELLISEARCH Backend" cmd /k "python main.py"
cd ..\..

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting React frontend on http://localhost:3000
cd web-app\frontend
start "INTELLISEARCH Frontend" cmd /k "npm run dev"
cd ..\..

echo.
echo 🚀 INTELLISEARCH Web Application is starting!
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/api/docs
echo.
echo Services are running in separate windows.
echo Close the command windows to stop the services.
echo.
pause