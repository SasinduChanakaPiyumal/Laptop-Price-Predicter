@echo off
REM Security Scans Execution Script for Windows
REM This script runs pip-audit and gitleaks scans to establish baseline security posture
REM Generates both JSON (human-readable) and SARIF (tool-integration) formats

echo ==========================================
echo Security Baseline Scans
echo ==========================================
echo.

REM Verify tools are installed
echo Verifying security tools installation...

where pip-audit >nul 2>nul
if %errorlevel% neq 0 (
    echo X pip-audit not found. Please run setup_security_tools.bat first.
    exit /b 1
)

where gitleaks >nul 2>nul
if %errorlevel% neq 0 (
    echo X gitleaks not found. Please run setup_security_tools.bat first.
    exit /b 1
)

echo √ pip-audit found
pip-audit --version

echo √ gitleaks found
gitleaks version

echo.

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo X requirements.txt not found. Cannot run pip-audit.
    exit /b 1
)

echo ==========================================
echo 1. Running pip-audit (JSON format)...
echo ==========================================
pip-audit --desc --format json --output pip-audit-results.json
if %errorlevel% neq 0 (
    echo X pip-audit JSON scan failed
    exit /b %errorlevel%
)
echo √ Generated: pip-audit-results.json
echo.

echo ==========================================
echo 2. Running pip-audit (SARIF format)...
echo ==========================================
pip-audit --format sarif --output pip-audit.sarif
if %errorlevel% neq 0 (
    echo X pip-audit SARIF scan failed
    exit /b %errorlevel%
)
echo √ Generated: pip-audit.sarif
echo.

echo ==========================================
echo 3. Running gitleaks (JSON format)...
echo ==========================================
gitleaks detect --no-git --verbose --report-path gitleaks-report.json
if %errorlevel% equ 1 (
    echo ! gitleaks detected potential secrets (exit code 1)
    echo √ Generated: gitleaks-report.json
) else if %errorlevel% neq 0 (
    echo X gitleaks failed with exit code %errorlevel%
    exit /b %errorlevel%
) else (
    echo √ Generated: gitleaks-report.json (no secrets detected)
)
echo.

echo ==========================================
echo 4. Running gitleaks (SARIF format)...
echo ==========================================
gitleaks detect --no-git --report-format sarif --report-path gitleaks.sarif
if %errorlevel% equ 1 (
    echo ! gitleaks detected potential secrets (exit code 1)
    echo √ Generated: gitleaks.sarif
) else if %errorlevel% neq 0 (
    echo X gitleaks failed with exit code %errorlevel%
    exit /b %errorlevel%
) else (
    echo √ Generated: gitleaks.sarif (no secrets detected)
)
echo.

echo ==========================================
echo Scan Summary
echo ==========================================
echo Output files generated:
echo   - pip-audit-results.json (vulnerability report)
echo   - pip-audit.sarif (SARIF format for tools)
echo   - gitleaks-report.json (secrets detection report)
echo   - gitleaks.sarif (SARIF format for tools)
echo.
echo Next steps:
echo 1. Review pip-audit-results.json for vulnerabilities
echo 2. Review gitleaks-report.json for detected secrets
echo 3. Document findings in SECURITY_SCAN_RESULTS.md
echo 4. Address any CRITICAL or HIGH severity issues
echo.
echo Note: These files are git-ignored and will not be committed.
echo ==========================================
