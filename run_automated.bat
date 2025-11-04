@echo off
REM run_automated.bat
REM Quick automated research with minimal setup
REM Prerequisites: Virtual environment must already exist

echo ===============================================
echo      INTELLISEARCH - Quick Automated Mode  
echo ===============================================
echo.
echo This script provides FAST automated research with:
echo   - No user prompts during workflow
echo   - Pre-configured settings (general prompt type)
echo   - Minimal setup checks
echo   - Unified report format (500-2000 words)
echo.
echo For first-time setup, use: run_interactive.bat
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please run setup first: run_interactive.bat
    echo.
    pause
    exit /b 1
)

REM Get query from user
set /p QUERY="Enter your research query: "

REM Default settings for automation
set PROMPT_TYPE=general

echo.
echo Configuration:
echo   Query: %QUERY%
echo   Mode: Automated (no user prompts)
echo   Prompt Type: %PROMPT_TYPE%
echo   Report Format: Unified (500-2000 words)
echo.

echo Starting automated research...
echo.

REM Run the automated workflow
.venv\Scripts\python.exe app.py "%QUERY%" --prompt-type %PROMPT_TYPE% --automation full

echo.
echo Automated research completed!
echo Check for generated report files in the current directory.
pause