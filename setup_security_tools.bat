@echo off
REM Security Tools Installation Script for Windows
REM This script installs pip-audit and gitleaks for security scanning

echo ==========================================
echo Security Tools Setup (Windows)
echo ==========================================
echo.

REM Install pip-audit
echo Installing pip-audit...
pip install pip-audit

REM Verify pip-audit installation
pip-audit --version >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] pip-audit installed successfully
    pip-audit --version
) else (
    echo [ERROR] pip-audit installation failed
    exit /b 1
)

echo.

REM Install gitleaks
echo Installing gitleaks...
echo.
echo Gitleaks for Windows needs to be installed manually:
echo.
echo 1. Visit: https://github.com/gitleaks/gitleaks/releases
echo 2. Download: gitleaks_8.18.1_windows_x64.zip (or latest version)
echo 3. Extract the zip file
echo 4. Move gitleaks.exe to a directory in your PATH
echo    Recommended: C:\Program Files\gitleaks\
echo 5. Add that directory to your PATH environment variable if needed
echo.
echo Alternative (using Chocolatey if installed):
echo    choco install gitleaks
echo.
echo Alternative (using Scoop if installed):
echo    scoop install gitleaks
echo.

REM Check if gitleaks is already installed
where gitleaks >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] gitleaks is already installed
    gitleaks version
) else (
    echo [INFO] gitleaks not found in PATH
    echo Please install gitleaks manually as described above
)

echo.
echo ==========================================
echo Setup Instructions Complete
echo ==========================================
echo.
echo Next steps:
echo 1. Verify pip-audit: pip-audit --version
echo 2. Install gitleaks manually (see instructions above)
echo 3. Verify gitleaks: gitleaks version
echo 4. See SECURITY.md for usage instructions

pause
