@echo off
REM Pre-commit Hook Installation Script for Windows
REM This script installs and configures pre-commit hooks for automated security scanning

echo ===================================
echo Pre-commit Hook Installation
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if pip is installed
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: pip is not installed
    echo Please install pip and try again
    pause
    exit /b 1
)

echo [OK] pip found
python -m pip --version
echo.

REM Install/upgrade pre-commit
echo Installing pre-commit...
python -m pip install --upgrade pre-commit

if %errorlevel% neq 0 (
    echo Error: Failed to install pre-commit
    pause
    exit /b 1
)

echo [OK] pre-commit installed successfully
echo.

REM Check if we're in a git repository
if not exist .git (
    echo Error: Not a git repository
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo [OK] Git repository detected
echo.

REM Install pre-commit hooks
echo Installing git hooks...
pre-commit install

if %errorlevel% neq 0 (
    echo Error: Failed to install git hooks
    pause
    exit /b 1
)

echo [OK] Git hooks installed successfully
echo.

REM Run pre-commit on all files as a test
echo Running initial test on all files...
echo This may take a few minutes on first run...
echo.

pre-commit run --all-files

REM Always show completion message
echo.
echo ===================================
echo Setup Complete!
echo ===================================
echo.
echo Pre-commit hooks are now active and will run automatically on:
echo   * git commit (blocks commits if checks fail)
echo.
echo Configured hooks:
echo   * gitleaks - Scans for secrets and credentials
echo   * pip-audit - Checks for vulnerable dependencies
echo.
echo To run manually: pre-commit run --all-files
echo To bypass hooks (emergency only): git commit --no-verify
echo.
echo For more information, see PRE_COMMIT_SETUP.md
echo.

pause
