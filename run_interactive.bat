@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo      INTELLISEARCH Setup & Interactive Mode
echo ===============================================
echo.
echo This script will:
echo   - Check Python installation
echo   - Create/activate virtual environment
echo   - Install required packages
echo   - Check for .env configuration
echo   - Run INTELLISEARCH in INTERACTIVE mode
echo.
echo NOTE: INTELLISEARCH now uses a unified report format (500-2000 words)
echo with query-answer structure and citations.
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo.
    pause
    exit /b 1
)

echo Python is installed:
python --version
echo.

:: Change to the script directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Check if requirements are installed by validating LangChain imports
echo Validating LangChain/LangGraph imports...
python -c "from src.import_validator import validate_imports; v = validate_imports(); v.print_status_report(); exit(0 if not v.get_missing_packages() else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages from requirements.txt...
    echo This may take a few minutes...
    
    :: Upgrade core tools first
    python -m pip install --upgrade pip setuptools wheel
    
    :: Install all packages from requirements.txt
    if exist "requirements.txt" (
        echo Installing packages from requirements.txt...
        pip install -r requirements.txt
    ) else (
        echo ERROR: requirements.txt file not found
        pause
        exit /b 1
    )
    
    echo.
    echo Validating imports after installation...
    python -c "from src.import_validator import validate_imports; v = validate_imports(); v.print_status_report()"
    
    echo.
    echo Package installation process completed.
    echo.
) else (
    echo All required packages are already installed.
    echo.
)

:: Check for .env file
if not exist ".env" (
    echo WARNING: .env file not found
    echo You may need to create a .env file with your API keys
    echo Example:
    echo SERPER_API_KEY=your_serper_key
    echo GOOGLE_API_KEY=your_google_key
    echo.
    echo Do you want to continue anyway? (y/n)
    set /p continue="Enter choice: "
    if /i "!continue!" neq "y" (
        echo Setup cancelled.
        pause
        exit /b 0
    )
    echo.
) else (
    echo .env file found - API keys will be loaded.
    echo.
)

:: Run the application
echo ===============================================
echo      Starting INTELLISEARCH (Interactive)
echo ===============================================
echo.
echo NOTE: You will be prompted to configure your research during execution.
echo Reports are now generated in a unified format with query-answer structure.
echo.

python app.py

:: Check if the application ran successfully
if %errorlevel% neq 0 (
    echo.
    echo ===============================================
    echo Application encountered an error (Exit Code: %errorlevel%)
    echo Check the output above for details.
    echo ===============================================
) else (
    echo.
    echo ===============================================
    echo Application completed successfully!
    echo Check for generated report files in the current directory.
    echo ===============================================
)

echo.
pause
