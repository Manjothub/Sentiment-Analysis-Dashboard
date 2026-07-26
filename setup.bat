@echo off
REM ============================================================================
REM Sentiment Analysis Dashboard - Windows Setup Script
REM ============================================================================
REM This script sets up the Python virtual environment and installs dependencies.
REM ============================================================================

echo ============================================================================
echo   Sentiment Analysis Dashboard - Setup Script
echo ============================================================================
echo.

REM Check Python installation
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM Create virtual environment
echo [Step 1/4] Creating Python virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
)
echo.

REM Activate virtual environment
echo [Step 2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

REM Upgrade pip
echo [Step 3/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [Step 4/4] Installing project dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
cd ..
echo [OK] Dependencies installed successfully.
echo.

REM Download NLTK data
echo [INFO] Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('vader_lexicon', quiet=True)"
echo [OK] NLTK data downloaded.
echo.

REM Download spaCy model
echo [INFO] Downloading spaCy model...
python -m spacy download en_core_web_sm --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Failed to download spaCy model. You can manually install it later.
) else (
    echo [OK] spaCy model downloaded.
)
echo.

REM Create required directories
echo [INFO] Creating required directories...
if not exist "dataset\processed" mkdir dataset\processed
if not exist "ml_models\checkpoints" mkdir ml_models\checkpoints
if not exist "ml_models\saved_models" mkdir ml_models\saved_models
if not exist "ml_models\training" mkdir ml_models\training
echo [OK] Directories created.
echo.

echo ============================================================================
echo   Setup Complete!
echo ============================================================================
echo.
echo To activate the virtual environment, run:
echo     venv\Scripts\activate
echo.
echo To run the dashboard:
echo     cd backend ^&^& python app.py
echo.
echo To run preprocessing:
echo     cd backend ^&^& python scripts/run_preprocessing.py
echo.
echo To run EDA:
echo     cd backend ^&^& python scripts/exploratory_analysis.py
echo.
pause
